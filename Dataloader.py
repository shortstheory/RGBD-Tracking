#Dataloader class

import numpy as np
import scipy.io as sio
from os.path import dirname, join as pjoin
import cv2

class Dataloader:
  
  def __init__(self, directory, dataset_name = "bear_front"):
    self.directory = directory
    self.mat_content = sio.loadmat(pjoin(self.directory, "frames.mat"))
    self.frames = self.mat_content["frames"]
    self.K = self.frames[0,0]['K']
    self.getIntrinsics()
    self.getNumOfFrames()
    self.getTimeandFrameIDs()
    self.bboxes = np.loadtxt(pjoin(directory, dataset_name + ".txt"), delimiter = ",")
    self.init_bbox = np.loadtxt(pjoin(directory, "init.txt"), delimiter = ",")


  def getIntrinsics(self):
    self.cx = self.K[0,2] 
    self.cy = self.K[1,2]  
    self.fx = self.K[0,0]
    self.fy = self.K[1,1]

  def getNumOfFrames(self):
    self.numOfFrames = self.frames[0,0]['length'][0][0]

  def getTimeandFrameIDs(self):
    self.imageTimeStamps = self.frames[0,0]['imageTimestamp'][0]
    self.imageFrameIDs = self.frames[0,0]['imageFrameID'][0]
    self.depthTimeStamps = self.frames[0,0]['depthTimestamp'][0]
    self.depthFrameIDs = self.frames[0,0]['depthFrameID'][0]

  def getRGB(self, frameId):
    imageName = pjoin(self.directory, "rgb/r-%d-%d.png" %(imageTimeStamps[frameId], imageFrameIDs[frameId]))
    img = cv2.imread(imageName)

    return img

  def getDepth(self, frameId):
    depthName = pjoin(self.directory, "depth/d-%d-%d.png" %(depthTimeStamps[frameId], depthFrameIDs[frameId]))
    depth = cv2.imread(depthName,0)
    depth = (depth << 3) | (depth << (16 - 3))
    depth_div = depth/1000

    return depth_div

  def getXYZ(self, frameId):
    img = self.getRGB(frameId)
    depth_div = self.getDepth(frameId)
    
    [x,y] = np.meshgrid(np.arange(0, 640), np.arange(0, 480))
    Xworld = ((x-cx)*depth_div)*1/fx
    Yworld = ((y-cy)*depth_div)*1/fy
    Zworld = depth_div
    XYZarray = np.dstack((Xworld, Yworld, Zworld))

    return XYZarray

  def getBbox(self, frameId):
    bbox_coords = self.bboxes[frameId]

    return bbox_coords