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
from BoundingBoxMessage import BoundingBoxMessage
import json
from BoxKalmanFilter import BoxKalmanFilter
from sort import Sort
class DepthTracker:
    def __init__(self):
        self.points_sub = rospy.Subscriber("/camera/depth/points",PointCloud2,self.pointCloudCallback,queue_size=1)
        self.kalmanInit = False
        self.bounding_box_sub = rospy.Subscriber("bounding_box",String,self.boundingBoxCallback,queue_size=1)
        self.rgb_img_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.imageCallback,queue_size=1, buff_size=2**24)
        self.xyz_array = np.zeros((480,640,3))
        self.boxKalmanFilter = []
        self.latestBbox = []
        self.mot_tracker = Sort()
        self.trackers = []
        self.counter = 0

    def boundingBoxCallback(self,boundingBoxMsgString):
        boundingBoxString = boundingBoxMsgString.data    #reads the boundingBox string dictionary
        boundingBoxData = json.loads(boundingBoxString)  #loads the dictionary
        bboxes = boundingBoxData['bounding_boxes']       #loads the boundingBox key data and bboxes[0] contains [X_topleft, Y_topleft, X_bottomright, Y_bottomright, confidence]
        bboxes_class = boundingBoxData['object_class']
        # print(bboxes_class)
        self.counter+=1
        # print("box condition: ", type(bboxes))
        # print("box len: ", len(bboxes))
        # if len(bboxes)>0:
            # if self.counter%1==0:
            #     self.trackers = self.mot_tracker.update(np.array(bboxes))
            # else:
            #     self.trackers = self.mot_tracker.update(np.array([]))


        if (len(bboxes) != 0 and bboxes_class == "person"):
            self.latestBbox = bboxes[0]                 #assuming when the filter starts, the object is in frame of view
            if self.kalmanInit == False:
                self.boxKalmanFilter = BoxKalmanFilter(bboxes[0], dt = 0.1)
                self.kalmanInit = True
        
        if self.kalmanInit == True:
            a = self.boxKalmanFilter.predict()
            # print("predicted_box: ",a)
            if (len(bboxes) != 0 and bboxes_class == "person"):                    #only update if there is a measurement.
                self.boxKalmanFilter.update(bboxes[0])

    def pointCloudCallback(self,cloud):
        self.xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud,remove_nans=False)

    def imageCallback(self,imageMsg):
        if self.kalmanInit == True:
            box = self.boxKalmanFilter.get_state()[0]
            # print("box: ", box)
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(imageMsg, desired_encoding='passthrough')
            (startX, startY, endX, endY) = box.astype("int")
            (bstartX, bstartY, bendX, bendY,confidence) = [int(i) for i in self.latestBbox]
            # if (len(self.trackers)>0):
            if (len(box)>0):
                # (SstartX, SstartY, SendX, SendY,_) = self.trackers[0].astype("int")
                # print("here")
                # print(startX, startY, endX, endY, "\n")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    np.array([200,0,0]), 2)
                # cv2.rectangle(frame, (bstartX, bstartY), (bendX, bendY),
                #     np.array([0,200,0]), 2)
                # cv2.rectangle(frame, (SstartX, SstartY), (SendX, SendY),
                #     np.array([0,0,200]), 2)                
                cv2.imshow('cv_img', frame)
                cv2.waitKey(2)

def main(args):
    rospy.init_node("DepthTrackerNode", anonymous=True)
    bbox = DepthTracker()
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)        