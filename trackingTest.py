# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
import time
import gc
from random import random


# %%
def runPrediction(dt,dl,T,R,p=1):
    deltaT = 1/30.0
    bestP = []
    gtP = []
    W = []
    error=0
    lastIdx = 0
    start = time.time()
    c = np.random.choice([0,1],p=[p,1-p])
    for i in tqdm(range(dl.bboxes.shape[0])):
        scanFrame = False
        if i==0:
            scanFrame = True
            xyz = dl.getXYZ(i)
        else:
            if c == 1:
                xyz = dl.getXYZ(i)
            else:
                xyz = np.empty((480,640,3))
                xyz[:,:,:] = np.nan
        img = dl.getRGB(i)
        bbox,_ = dl.getBbox(i)
        bestParticle,idx = dt.updateMeasurements(img,xyz,bbox,T,R,deltaT,scanFrame)
        w = np.sum(dt.particleFilter.weights*dt.particleFilter.weights)
        W.append((1/w)/particleN)
        if bestParticle is not None:
            lastIdx = idx
            bestP.append(bestParticle)
        else:
            bestP.append(dt.particleFilter.particles[lastIdx])
            
        if np.all(bbox[0]>0) == True:
            gt_xyz = dl.getXYZ(i)
            u,v = dt.vision.getBBoxCenter(bbox[0])
            gt_origin = gt_xyz[u,v,:]
            error += np.linalg.norm(bestP[-1][:3]-gt_origin)
            gtP.append(gt_origin)
        else:
            gtP.append([None,None,None])
    end = time.time()
    deltaT = end-start
    return (bestP,gtP,W,error,deltaT)

T = np.array([0,0,0])
R = np.eye(3)
particleN = 100
particleCov = 0.001*np.diag([.01,.01,.01,.001,.001,.001])
dl = Dataloader('/home/nic/ms-work/dtplayground/bear_front/')

# %%
dt = DepthTracker(dl.K,particleN,particleCov,0.6,'sift')
siftData=runPrediction(dt,dl,T,R,1)
dt = DepthTracker(dl.K,particleN,particleCov,0.6,'surf')
surfData=runPrediction(dt,dl,T,R,1)
dt = DepthTracker(dl.K,particleN,particleCov,0.6,'orb')
orbData=runPrediction(dt,dl,T,R,1)

# %%
accRuns = []
for i in tqdm(range(100)):
    dt = DepthTracker(dl.K,particleN,particleCov,0.6,'sift')
    runData=runPrediction(dt,dl,T,R,i/100)
    accRuns.append(runData)

# %%
siftPlot = np.array(siftData[0])[:,:3]
surfPlot = np.array(surfData[0])[:,:3]
orbPlot = np.array(orbData[0])[:,:3]
gtPlot = np.array(siftData[1])[:,:3]
fig, ax = plt.subplots(1,4,figsize=(18, 4))
# fig.suptitle('Feature Descriptor Comparison\n')
ax[0].plot(gtPlot)
ax[1].plot(siftPlot)
ax[2].plot(surfPlot)
ax[3].plot(orbPlot)
ax[0].set_title('Ground Truth',fontsize=18)
ax[1].set_title('SIFT',fontsize=18)
ax[2].set_title('SURF',fontsize=18)
ax[3].set_title('ORB',fontsize=18)
ax[0].set_ylabel('Meters',fontsize=12)
ax[1].set_xlabel('Runtime: '+str(round(siftData[4],3))+' secs\nMSE: '+str(round(siftData[3]/len(gtPlot),3))+' m',fontsize=12)
ax[2].set_xlabel('Runtime: '+str(round(surfData[4],3))+' secs\nMSE: '+str(round(surfData[3]/len(gtPlot),3))+' m',fontsize=12)
ax[3].set_xlabel('Runtime: '+str(round( orbData[4],3))+' secs\nMSE:'+str(round( orbData[3]/len(gtPlot),3) )+' m',fontsize=12)
fig.legend(ax,labels=['X','Y','Z'],
           loc="top right",
           borderaxespad=2,
           )
fig.savefig('descriptors.eps',bbox_inches='tight')
# %%
siftPlot = np.array(siftData[0])[:,3:6]
surfPlot = np.array(surfData[0])[:,3:6]
orbPlot =   np.array(orbData[0])[:,3:6]
gtPlot =   np.array(siftData[1])[:,3:6]
fig, ax = plt.subplots(1,3,figsize=(16, 4))
# fig.suptitle('Feature Descriptor Comparison\n')
ax[0].plot(siftPlot)
ax[1].plot(surfPlot)
ax[2].plot(orbPlot)
ax[0].set_title('SIFT',fontsize=18)
ax[1].set_title('SURF',fontsize=18)
ax[2].set_title('ORB',fontsize=18)
# ax[0].set_xlabel('Runtime: '+str(round(siftData[4],3))+' secs\nMSE: '+str(round(siftData[3]/len(gtPlot),3))+' m',fontsize=12)
# ax[1].set_xlabel('Runtime: '+str(round(surfData[4],3))+' secs\nMSE: '+str(round(surfData[3]/len(gtPlot),3))+' m',fontsize=12)
# ax[2].set_xlabel('Runtime: '+str(round( orbData[4],3))+' secs\nMSE:'+str(round( orbData[3]/len(gtPlot),3) )+' m',fontsize=12)
fig.legend(ax,labels=['Vx','Vy','Vz'],
           loc="top right",
           borderaxespad=2,
           )
fig.savefig('velocities.eps',bbox_inches='tight')
# %%
import pickle
output = open('descs.pkl', 'wb')
pickle.dump((siftData,surfData,orbData), output)
output.close()

# %%
nparticles = [100,500,1000]
data = []
for p in nparticles:
    dt = DepthTracker(dl.K,p,particleCov,0.6,'sift')
    data.append(runPrediction(dt,dl,T,R,1))


# %%
import pickle
output = open('data1005001000.pkl', 'wb')
pickle.dump(data, output)
output.close()

# %%
import pickle
with open('data51050.pkl','rb') as f:
    d1 = pkl.load(f)
with open('data1005001000.pkl','rb') as f:
    d2 = pkl.load(f)


# %%
nparticles = [5,10,50,100,500,1000]

d = d1+d2
t = []
err = []
for _d in d:
    err.append(_d[3]/len(d[0][0]))
    t.append(_d[4])

fig, ax = plt.subplots(1,3,figsize=(16, 4))
ax[0].plot(nparticles,err)
ax[0].set_xlabel('Particles',fontsize=12)
ax[0].set_ylabel('MSE (m)',fontsize=12)
ax[0].set_title('Particles vs MSE',fontsize=18)
ax[1].plot(nparticles,t)
ax[1].set_xlabel('Particles',fontsize=12)
ax[1].set_ylabel('Runtime (s)',fontsize=12)
ax[1].set_title('Particles vs Runtime',fontsize=18)
ax[2].plot(dropRate,errD)
ax[2].set_xlabel('Drop Rate',fontsize=12)
ax[2].set_ylabel('MSE (m)',fontsize=12)
ax[2].set_title('Drop Rate vs MSE',fontsize=18)

fig.savefig('combined.eps',bbox_inches='tight')
# %%
dropRate = [0,0.25,0.5,0.75,1]
dataDrop = []
for dr in dropRate:
    dt = DepthTracker(dl.K,100,particleCov,0.6,'sift')
    dataDrop.append(runPrediction(dt,dl,T,R,dr))
import pickle
output = open('dropdata.pkl', 'wb')
pickle.dump(dataDrop, output)
output.close()


# %%
dropRate = [0,0.25,0.5,0.75,1]

import pickle
with open('dropdata.pkl','rb') as f:
    dataDrop = pkl.load(f)
t = []
errD = []
for _d in dataDrop:
    errD.append(_d[3]/len(_d[0]))
    t.append(_d[4])
fig, ax = plt.subplots(1,1,figsize=(4, 4))

ax.plot(dropRate,err)
ax.set_xlabel('Drop Rate')
ax.set_ylabel('MSE')
fig.savefig('droprate.eps',bbox_inches='tight')
# %%
