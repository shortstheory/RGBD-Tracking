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
from ParticleFilter import PF
from Helpers import Keypoints3D
from Vision import Vision
#ROS Imports
import os
import copy
import pickle as pkl

class DepthTracker:
    def __init__(self, cameraK, cameraR, h, w):
        self.objectModel = Keypoints3D()
        self.xyz_array = np.zeros((480,640,3))
        self.img = np.zeros((480,640,3))
        self.bbox = [0,0,0,0]

        self.vision = Vision(cameraK, cameraR)
        self.particleFilter = None

    def updateBBox(self, bboxes):
        # have a score function here for checking which bounding box tracks the best based on dt
        if len(bboxes)==1:
            self.bbox = bboxes[0]
        elif len(bboxes)>1:
            max_iou = -1
            max_idx = -1
            for i,bbox in enumerate(bboxes):
                iou = self.vision.IOU(bbox, self.bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i
            self.bbox = bboxes[max_idx]

    def updateMeasurements(self, img, xyz_array, bboxes, cameraT, cameraR, dt, scanFrame=False):
        self.xyz_array = xyz_array
        self.latestImg = img
        self.updateBBox(bboxes)

        if scanFrame==True:
            self.objectModel, origin3D = self.vision.scanObject(self.img, self.bbox, self.xyz_array)
            self.particleFilter = PF(origin3D)
        else:
            # need to handle cases here for pf
            # assume bbox is always in image frame (but need not be in RGB)
            self.particleFilter.update(dt)
            if self.bbox is not None:
                keypoints, descriptors, img = self.vision.getKeypoints2D(img, self.bbox)
                match, dMatch = self.vision.featureMatch(self.objectModel.descriptors, descriptors)
                for particle in self.particleFilter.particles:
                    particleXYZ = particle[:3]
                    u,v = self.vision.getBBoxCenter(self.bbox)
                    origin = self.xyz_array[u,v,:]
                    if np.isnan(origin).any() == False:
                        # evaluate all particles on 3D correlations EASY
                        print('')
                    else:
                        # evaluate on 2D HARD
                        cameraPoints3D = self.vision.globalKeypoint2camera(match, self.objectModel, particleXYZ, \
                                                                        cameraT, cameraR)
                        pixels = self.vision.camera2pixel(cameraPoints3D)
                        print('')
                    # send correlations to filter

    # pass keypoints to this so we dont need to do SIFT over and over again
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

#%%
def main(args):
    dt = DepthTracker(None,None,None,None)

if __name__=='__main__':
	main(sys.argv)