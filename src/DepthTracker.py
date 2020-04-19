#!/usr/bin/env python2
from __future__ import print_function
import sys
import math
import numpy as np
import tf
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
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
        self.tf = tf.TransformListener()
        self.odomFrame = "odom"
        self.cameraOpticalFrameRGB = "camera_color_optical_frame"

    def boundingBoxCallback(self,boundingBoxMsgString):
        boundingBoxString = boundingBoxMsgString.data
        boundingBoxData = json.loads(boundingBoxString)
        bboxes = boundingBoxData['bounding_boxes']
        if len(bboxes)>0:
            self.latestBbox = bboxes[0]

    def pointCloudCallback(self,cloud):
        self.xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud,remove_nans=False)

    def imageCallback(self,imageMsg):
        frame = self.bridge.imgmsg_to_cv2(imageMsg, desired_encoding='passthrough')

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
        campoints3D = self.xyz_array[startY:endY,startX:endX,:]
        origin3D = campoints3D[cvimg.shape[0]//2,cvimg.shape[1]//2,:]
        keypoints3D = Keypoints3D()
        for kp,desc in zip(keypoints,descriptors):
            # find a way to interpolate depth map
            u,v = int(kp.pt[1]), int(kp.pt[0])
            point3D = campoints3D[u,v]-origin3D
            if np.isnan(point3D).any() == False:
                keypoints3D.add(kp,desc,point3D)
        imgX = cv2.drawKeypoints(cvimg,keypoints,None)
        cv2.imshow('cv_imgX', imgX)
        cv2.waitKey(2)
        return keypoint3DList

    # pixels is a 3xN array
    def pixel2camera(self,pixels,z_estimate):
        # python2 doesnt support @ :(
        # make sure units of z_estd are correct
        cameraPoints3D = z_estimate*np.matmul(np.linalg.inv(self.cameraK),pixels)
        return cameraPoints3D
    
    def getTfTransform(self, destination, source):
        if self.tf.frameExists(destination) and self.tf.frameExists(source):
            t = self.tf.getLatestCommonTime(destination, source)
            return self.tf.lookupTransform(destination, source, t)
        return None, None

    def processFrame(self, img, bbox):
        keypoints, descriptors, img = self.getKeypoints2D(img, bbox)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(self.objectModel.descriptors,descriptors,k=2)
        # Need to draw only good matches, so create a mask
        match = []
        dMatch = []
        for d1,d2 in matches:
            if d1.distance < 0.7*d2.distance:
                match.append(d1.trainIdx)
                dMatch.append(d1)
            else:
                match.append(-1)
        return match,dMatch

    def keypoint2camera(self, keypointMatches, particleGlobalXYZ):
        translation, rotation = self.getTfTransform(self.cameraOpticalFrameRGB,self.odomFrame)
        Hcw = self.tf.fromTranslationRotation(translation,rotation)
        points3D = np.zeros((4,np.sum(keypointMatches)))
        points3D[3,:] = np.ones((np.sum(keypointMatches)))
        for i in range(len(keypointMatches)):
            if keypointMatches[i] == 1:
                points3D[0:3,i] = self.objectModel[i].worldPointXYZ+particleGlobalXYZ
        cameraPoints3D = np.matmul(Hcw,points3D)
        cameraPoints3D = cameraPoints3D/cameraPoints3D[-1,:]
        return cameraPoints3D
        #convert this to camera frame


def main(args):
    rospy.init_node("DepthTrackerNode", anonymous=True)
    bbox = DepthTracker()
    listener = tf.TransformListener()
    
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)