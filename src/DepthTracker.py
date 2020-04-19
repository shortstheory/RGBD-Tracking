#!/usr/bin/env python2
from __future__ import print_function
import sys
import math
import numpy as np
# import tf
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
        self.latestImg = np.zeros((640,480,3))
        self.bridge = CvBridge()
        self.sift = cv2.xfeatures2d.SIFT_create()

    def boundingBoxCallback(self,boundingBoxMsgString):
        boundingBoxString = boundingBoxMsgString.data
        boundingBoxData = json.loads(boundingBoxString)
        bboxes = boundingBoxData['bounding_boxes']
        if len(bboxes)>0:
            self.latestBbox = bboxes[0]
            self.scanObject(self.latestImg,self.latestBbox,self.xyz_array)

    def pointCloudCallback(self,cloud):
        self.xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud,remove_nans=False)

    def imageCallback(self,imageMsg):
        frame = self.bridge.imgmsg_to_cv2(imageMsg, desired_encoding='passthrough')
        self.latestImg = copy.deepcopy(frame)
        (startX, startY, endX, endY,_) = [int(i) for i in self.latestBbox]
        cv2.rectangle(frame, (startX, startY), (endX, endY), np.array([0,200,0]), 2)
        # cv2.imshow('cv_img', frame)
        # cv2.waitKey(2)

    def scanObject(self, cvimg, bbox, xyz_array):
        # sift ignores colour, so convert to grayscale
        (startX, startY, endX, endY,_) = [int(i) for i in bbox]
        cvimg = cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)

        cvimg = cvimg[startY:endY,startX:endX]
        campoints3D = self.xyz_array[startY:endY,startX:endX,:]

        keypoints, descriptors = self.sift.detectAndCompute(cvimg,None)
        origin = np.array([cvimg.shape[0]//2,cvimg.shape[1]//2])
        origin3D = campoints3D[cvimg.shape[0]//2,cvimg.shape[1]//2,:]
        keypoint3DList = []
        print(campoints3D.shape)
        print(origin3D)
        for kp,desc in zip(keypoints,descriptors):
            # find a way to interpolate depth map
            u,v = int(kp.pt[1]), int(kp.pt[0])
            point3D = campoints3D[u,v]-origin3D
            if np.isnan(point3D).any() == False:
                keypoint3DList.append(Keypoint3D(kp,desc,point3D))
        print(len(keypoint3DList))
        imgX = cv2.drawKeypoints(cvimg,keypoints,None)
        # for kp3d in keypoint3DList:
        #     print(kp3d.point3D)
        cv2.imshow('cv_imgX', imgX)
        cv2.waitKey(2)
        return keypoint3DList




def main(args):
    rospy.init_node("DepthTrackerNode", anonymous=True)
    bbox = DepthTracker()
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)        