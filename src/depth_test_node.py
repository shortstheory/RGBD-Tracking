#!/usr/bin/env python2
from __future__ import print_function
import sys
import math
import numpy as np
# import tf
# from cv_bridge import CvBridge, CvBridgeError
import cv2
# import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
#ROS Imports
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
import ros_numpy
import imutils
import os
from cv_bridge import CvBridge

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
proto = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
print(os.getcwd())
net = cv2.dnn.readNetFromCaffe(proto, model)


class DepthTest:
    def __init__(self):
        a=0
        self.points_sub = rospy.Subscriber("/camera/depth/points",PointCloud2,self.handlePc2,queue_size=1)
        self.rgb_img_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.handleImage,queue_size=1)
        self.xyz_array = np.zeros((480,640,3))
    def handlePc2(self,cloud):
        self.xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud,remove_nans=False)
        # print(xyz_array[240,320,:])
    def handleImage(self,msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame = imutils.resize(img, width=640)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # print(detections.shape)
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > .2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                midptX,midptY = (startX+endX)//2,(startY+endY)//2
                # print(midptX,midptY)
                print(self.xyz_array[midptY,midptX,:])
                # print(frame.shape)
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        # cv2.imshow('cv_img', frame)
        # cv2.waitKey(2)

def main(args):
    rospy.init_node("depth_test_node", anonymous=True)
    dt = DepthTest()
    # rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
