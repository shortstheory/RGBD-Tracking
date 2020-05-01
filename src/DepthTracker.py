#!/usr/bin/env python2
#%%
from __future__ import print_function
import sys
import math
import numpy as np
import tf
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import matplotlib
import matplotlib.pyplot as plt

#ROS Imports
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
import ros_numpy
# import imutils
from std_msgs.msg import String
import os
import json
from Helpers import *
import copy
import pickle as pkl
class DepthTracker:
    def __init__(self):
        self.points_sub = rospy.Subscriber("/camera/depth/points",PointCloud2,self.pointCloudCallback,queue_size=1)
        self.bounding_box_sub = rospy.Subscriber("bounding_box",String,self.boundingBoxCallback,queue_size=1)
        self.rgb_img_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.imageCallback,queue_size=1, buff_size=2**24)
        self.xyz_array = np.zeros((480,640,3))
        self.latestBbox = [0,0,0,0,0]
        self.latestImg = np.zeros((480,640,3))
        self.bridge = CvBridge()
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.cameraK = np.matrix([[462.1379497504639, 0.0, 320.5],[0.0, 462.1379497504639, 240.5],[0.0, 0.0, 1.0]])
        self.objectModel = Keypoints3D()
        # self.tf = tf.TransformListener()
        self.odomFrame = "odom"
        self.cameraOpticalFrameRGB = "camera_color_optical_frame"
        self.particles3D = np.zeros((1,3))
        self.cameraR = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

    def boundingBoxCallback(self,boundingBoxMsgString):
        boundingBoxString = boundingBoxMsgString.data
        boundingBoxData = json.loads(boundingBoxString)
        bboxes = boundingBoxData['bounding_boxes']
        if len(bboxes)>0:
            self.latestBbox = bboxes[0]
            with open('bbox.pkl', 'wb') as pickle_file:
                pkl.dump(self.latestBbox, pickle_file)

    def pointCloudCallback(self,cloud):
        self.xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud,remove_nans=False)
        with open('xyz.pkl', 'wb') as pickle_file:
            pkl.dump(self.xyz_array, pickle_file)

    def imageCallback(self,imageMsg):
        frame = self.bridge.imgmsg_to_cv2(imageMsg, desired_encoding='passthrough')
        cv2.imwrite('myman1.png',frame)

    def getKeypoints2D(self, img, bbox):
        (startX, startY, endX, endY,_) = [int(i) for i in bbox]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[startY:endY,startX:endX]
        keypoints, descriptors = self.sift.detectAndCompute(img,None)
        return keypoints, descriptors, img

    def scanObject(self, cvimg, bbox, xyz_array):
        # sift ignores colour, so convert to grayscale
        keypoints, descriptors, cvimg = self.getKeypoints2D(cvimg, bbox)
        origin = np.array([cvimg.shape[0]//2,cvimg.shape[1]//2])
        (startX, startY, endX, endY,_) = [int(i) for i in bbox]        
        campoints3D = xyz_array[startY:endY,startX:endX,:]
        origin3D = campoints3D[cvimg.shape[0]//2,cvimg.shape[1]//2,:]
        keypoints3D = Keypoints3D()
        for kp,desc in zip(keypoints,descriptors):
            # find a way to interpolate depth map
            u,v = int(kp.pt[1]+startY), int(kp.pt[0]+startX)
            point3D = np.matmul(self.cameraR.T,xyz_array[u,v,:]-origin3D)
            # print(point3D,campoints3D[u,v,:]-origin3D)
            if np.isnan(point3D).any() == False:
                point3D = np.array([0,0,0])
            keypoints3D.add(kp,desc,point3D)
                # print(campoints3D[u,v,:])
        keypoints3D.numpyify()
        self.objectModel = keypoints3D
        return keypoints3D,origin3D

    # pixels is a 3xN array
    def pixel2camera(self,pixels,z_estimate):
        # python2 doesnt support @ :(
        # make sure units of z_estd are correct
        cameraPoints3D = z_estimate*np.matmul(np.linalg.inv(self.cameraK),pixels)
        return cameraPoints3D

    def camera2pixel(self,cameraPoints3D):
        pixels = np.matmul(self.cameraK,(cameraPoints3D/cameraPoints3D[-1,:]))[:2]
        return pixels

    def getTfTransform(self, destination, source):
        if self.tf.frameExists(destination) and self.tf.frameExists(source):
            t = self.tf.getLatestCommonTime(destination, source)
            return self.tf.lookupTransform(destination, source, t)
        return None, None

    def processFrame(self, img, bbox, particles3D):
        keypoints, descriptors, img = self.getKeypoints2D(img, bbox)
        # Need to draw only good matches, so create a mask
        # each index of match is the pt in 2nd frame
        match, dMatch = self.featureMatch(self.objectModel.descriptors,descriptors)
        for particle in particles3D:
            cameraPoints3D = self.globalKeypoint2camera(match,particle)
            pixels = self.camera2pixel(cameraPoints3D)
        # do some correlation thing here

    def featureMatch(self, descs1, descs2):
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs1,descs2, k=2)
        match = []
        dMatch = []
        for d1,d2 in matches:
            if d1.distance < 0.7*d2.distance:
                match.append(1)
                dMatch.append(d1)
            else:
                match.append(0)
        return np.array(match),dMatch

    # leave the keypoints as they are. Instead, divide the camera points by z_estd
    def globalKeypoint2camera(self, keypointMatches, particleGlobal3D, T, R):
        # translation, rotation = self.getTfTransform(self.cameraOpticalFrameRGB,self.odomFrame)
        # Hcw = self.tf.fromTranslationRotation(translation,rotation)
        Hcw = np.zeros((4,4))
        Hcw[:3,:3] = R
        Hcw[:3,3] = T
        Hcw[3,3] = 1
        points3D = np.zeros((4,np.sum(keypointMatches)))
        points3D[3,:] = np.ones((np.sum(keypointMatches)))
        p3Didx = 0
        for k,idx in enumerate(keypointMatches):
            if idx != 0:
                points3D[0:3,p3Didx] = self.objectModel.objectPoints[k]+particleGlobal3D
                p3Didx += 1
                # print(self.objectModel.objectPoints[k])
        # print(points3D)
        cameraPoints3D = np.matmul(Hcw,points3D)
        cameraPoints3D = (cameraPoints3D/cameraPoints3D[-1,:])[:3,:]
        return cameraPoints3D
        #convert this to camera frame
#%%
def main(args):
    rospy.init_node("DepthTrackerNode", anonymous=True)
    bbox = DepthTracker()
    listener = tf.TransformListener()
    
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)