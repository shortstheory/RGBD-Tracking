#%%
from __future__ import print_function
import sys
import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from .ParticleFilter import PF
from .Helpers import Keypoints3D
from .Vision import Vision
#ROS Imports
import os
import copy
import pickle as pkl

'''
Class which implements RGB-D tracking
Takes intrinsic matrix, number of particles, covariance, threshold for resampling
Can use 'sift', 'surf', 'orb' for descriptors
'''
class DepthTracker:
    def __init__(self, cameraK, particleN, particleCov, pThresh=0.6, detector='sift'):
        self.objectModel = Keypoints3D()
        self.bbox = [0,0,0,0]

        self.vision = Vision(cameraK, detector)
        self.particleFilter = None
        self.particleN = particleN
        self.particleCov = particleCov
        self.pThresh = pThresh

    def updateBBox(self, bboxes):
        '''
        Finds the bounding box matching that from the previous frame using IOU
        '''
        if len(bboxes)==1:
            if np.all(bboxes[0]>0) == True:
                self.bbox = bboxes[0]
            else:
                self.bbox = None
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
        '''
        Updates particles based on keypoint matching or depth of the center of the
        bounding box if available
        '''
        self.updateBBox(bboxes)
        if scanFrame==True:
            self.objectModel, origin3D = self.vision.scanObject(img, self.bbox, xyz_array, cameraT, cameraR)
            pfInitialState = np.zeros(6)
            pfInitialState[:3] = origin3D
            self.particleFilter = PF(pfInitialState,self.particleN,self.particleCov)
        else:
            w = np.sum(self.particleFilter.weights*self.particleFilter.weights)
            effectiveParticleRatio = (1/w)/self.particleN
            if effectiveParticleRatio < self.pThresh:
                self.particleFilter.restratified_sampling()
            # assume bbox is always in image frame (but need not be in RGB)
            self.particleFilter.predict(dt)
            if self.bbox is not None:
                keypoints, descriptors, _ = self.vision.getKeypoints2D(img, self.bbox)
                match, dmatch = self.vision.featureMatch(self.objectModel.descriptors, descriptors)
                correlations = []
                if (np.sum(match)>=4):
                    for particle in self.particleFilter.particles:
                        u,v = self.vision.getBBoxCenter(self.bbox)
                        origin = xyz_array[u,v,:]
                        if np.isnan(origin).any() == False:
                            corr = self.correlation3D(particle, xyz_array, cameraT, cameraR)
                            correlations.append(corr)
                        else:
                            corr = self.correlation2D(particle, keypoints, descriptors, match, dmatch, cameraT, cameraR)
                            correlations.append(corr)
                    correlations = np.array(correlations)
                    bestParticleIdx = np.argmax(correlations)
                    self.particleFilter.update(correlations)
                    return self.particleFilter.particles[bestParticleIdx], bestParticleIdx
        return None, None

    def correlation2D(self, particle, kps, desc, match, dmatch, cameraT, cameraR):
        '''
        Computes the correlation when depth information is no longer available using
        keypoints and descriptors
        '''
        cameraPoints3D = self.vision.globalKeypoint2camera(match,self.objectModel,particle[:3],cameraT,cameraR)
        predictedKeypointPixels = self.vision.camera2pixel(cameraPoints3D)
        shiftedKeypointPixels = self.vision.shiftKeypoints(kps,dmatch,self.bbox)
        meanErr = np.linalg.norm(shiftedKeypointPixels-predictedKeypointPixels)/shiftedKeypointPixels.shape[1]
        return 1/meanErr
    
    def correlation3D(self, particle, xyz_array, cameraT, cameraR):
        '''
        Computes the correlation when depth information is avaialble
        '''
        u,v = self.vision.getBBoxCenter(self.bbox)
        pose = xyz_array[u,v,:]
        particle3D = particle[:3]
        particle3D = particle3D.reshape(3,1)
        cameraPoint = self.vision.transformPoints(cameraT, cameraR, particle3D).reshape(-1)
        pointErr = np.linalg.norm(pose-cameraPoint)
        return 1/pointErr