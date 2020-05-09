# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import numpy as np
import pickle as pkl
import cv2
from Dataloader import Dataloader
from src.Vision import Vision
from src.DepthTracker import DepthTracker
from tqdm import tqdm



# %%
particleN = 10
particleCov = 0.001*np.diag([.01,.01,.01,.001,.001,.001])
# dl = Dataloader('/home/nic/ms-work/dtplayground/bear_front/')
# dl = Dataloader('/home/nic/ms-work/dtplayground/new_ex_occ4/','new_ex_occ4')
dl = Dataloader('/home/nic/ms-work/dtplayground/face_occ5/','face_occ5')

bbox,_ = dl.getBbox(0)
# static camera
T = np.array([0,0,0])
R = np.eye(3)

dt = DepthTracker(dl.K,particleN,particleCov)
deltaT = 1/30.0
print(dl.numOfFrames)


# %%
bestP = []
gtP = []
error=0
lastIdx = 0
for i in tqdm(range(dl.bboxes.shape[0])):
    scanFrame = False
    if i==0:
        scanFrame = True
        xyz = dl.getXYZ(i)
    # if i < 120:
    #     xyz = dl.getXYZ(i)
    else:
        xyz = np.empty((480,640,3))
        xyz[:,:,:] = np.nan
    img = dl.getRGB(i)
    bbox,_ = dl.getBbox(i)
    bestParticle,idx = dt.updateMeasurements(img,xyz,bbox,T,R,deltaT,scanFrame)

    if bestParticle is not None:
        gt_xyz = dl.getXYZ(i)
        lastIdx = idx
        u,v = dt.vision.getBBoxCenter(bbox[0])
        gt_origin = gt_xyz[u,v,:]
        bestP.append(bestParticle)
        gtP.append(gt_origin)
        error += np.linalg.norm(bestParticle[:3]-gt_origin)
    else:
        bestP.append(dt.particleFilter.particles[lastIdx])
        gtP.append(np.array([0,0,0]))


# %%
print(error/dl.bboxes.shape[0])
bestP = np.array(bestP)
plt.plot(bestP[:,:3])


# %%
gtP = np.array(gtP)
# plt.scatter(np.arange(gtP.shape[0]),gtP)#,linestyle="None")#-bestP[:,:3])
plt.plot(gtP)


# %%
plt.imshow(dl.getDepth(0))


# %%
np.max(dl.getDepth(0))


# %%
depthname = '/home/nic/ms-work/dtplayground/bear_front/depth/d-0-1.png'
depth = cv2.imread(depthname,-1)
depth = (depth >> 3) | (depth << (16 - 3))
depth=depth/1000


# %%
np.max(depth)


# %%
np.diag(np.array([0.1,0.001,1]))


# %%
np.eye(3)


# %%


