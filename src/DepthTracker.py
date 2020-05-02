#!/usr/bin/env python2
#%%
from __future__ import print_function
import sys
import math
import numpy as np
import tf
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from particle_filter import PF
from Helpers import Keypoints3D
from Vision import Vision
#ROS Imports
import os
import copy
import pickle as pkl

class DepthTracker:
    def __init__(self, cameraK, cameraR, h, w):
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.objectModel = Keypoints3D()
        self.xyz_array = np.zeros((480,640,3))
        self.img = np.zeros((480,640,3))
        self.currentBbox = [0,0,0,0]

        self.vision = Vision(cameraK, cameraR)

    def updateBBox(self, bboxes):
        # have a score function here for checking which bounding box tracks the best based on dt
        if len(bboxes)==1:
            self.currentBbox = bboxes[0]
        elif len(bboxes)>1:
            max_iou = -1
            max_idx = -1
            for i,bbox in enumerate(bboxes):
                iou = self.vision.IOU(bbox, self.currentBbox)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i
            self.currentBbox = bboxes[max_idx]

    def updateMeasurements(self, img, xyz_array, bboxes):
        self.xyz_array = xyz_array
        self.latestImg = img


    def correlation2D(self, particle, img, bbox, cameraT, cameraR):
        # we can weight it by number of matches too
        kps, desc, img = self.vision.getKeypoints2D(img,bbox)
        match, dmatch = self.vision.featureMatch(self.objectModel.descriptors,desc)
        cameraPoints3D = self.vision.globalKeypoint2camera(match,particle,cameraT,cameraR)
        predictedKeypointPixels = self.vision.camera2pixel(cameraPoints3D)
        shiftedKeypointPixels = self.vision.shiftKeypoints(kps,dmatch,bbox)
        return np.linalg.norm(shiftedKeypointPixels-predictedKeypointPixels)
    
    def correlation3D(self, particle, img, bbox):
        u,v = int(bbox[1]+bbox[3])//2,int(bbox[0]+bbox[2])//2
        pose = self.xyz_array[u,v,:]
        return np.linalg.norm(pose-particle)

    def processFrame(self, img, bbox, particles3D):
        keypoints, descriptors, img = self.vision.getKeypoints2D(img, bbox)
        # Need to draw only good matches, so create a mask
        # each index of match is the pt in 2nd frame
        match, dMatch = self.vision.featureMatch(self.objectModel.descriptors,descriptors)
        for particle in particles3D:
            cameraPoints3D = self.vision.globalKeypoint2camera(match,particle)
            pixels = self.vision.camera2pixel(cameraPoints3D)
        # do some correlation thing here


        #convert this to camera frame
#%%
def main(args):
    dt = DepthTracker()

if __name__=='__main__':
	main(sys.argv)