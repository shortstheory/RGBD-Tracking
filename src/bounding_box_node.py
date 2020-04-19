#!/usr/bin/env python3
from __future__ import print_function
import sys
import math
import numpy as np
# import tf
# from cv_bridge import CvBridge, CvBridgeError
import cv2
import json
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
from Helpers import BoundingBoxMessage
from cv_bridge import CvBridge
from std_msgs.msg import String
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
proto = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
print(os.getcwd())
net = cv2.dnn.readNetFromCaffe(proto, model)

class BoundingBox:
    def __init__(self):
        self.rgb_img_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.handleImage,queue_size=1, buff_size=2**24)
        self.xyz_array = np.zeros((480,640,3))
        self.bounding_box_pub = rospy.Publisher('bounding_box',String,queue_size=1)

    def handleImage(self,msg):
        bboxList = []
        bboxMsg = BoundingBoxMessage()

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
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > .2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = np.append(box,confidence)
                bboxList.append(box.tolist())
                objclass = CLASSES[idx]
        bboxMsg.timestamp = msg.header.stamp.secs+10**(-9)*msg.header.stamp.nsecs
        bboxMsg.seq = msg.header.seq
        bboxMsg.bounding_boxes = bboxList
        self.bounding_box_pub.publish(json.dumps(bboxMsg.__dict__))

def main(args):
    rospy.init_node("bounding_box_node", anonymous=True)
    bbox = BoundingBox()
    # rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
