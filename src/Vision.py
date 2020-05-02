import cv2
import numpy as np
from Helpers import Keypoints3D

class Vision:
    def __init__(self,K,R):
        self.cameraK = K
        self.cameraR = R
        self.featureDetector = cv2.xfeatures2d.SIFT_create()
    
    def IOU(self, bbox, latestBbox):
        dx = min(bbox[2],latestBbox[2])-max(bbox[0],latestBbox[0])
        dy = min(bbox[3],latestBbox[3])-max(bbox[1],latestBbox[1])
        int_area = dx*dy
        area_1 = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        area_2 = (latestBbox[2]-latestBbox[0])*(latestBbox[3]-latestBbox[1])
        union_area = area_1+area_2-int_area
        iou = int_area/union_area
        return iou

    def getBBoxCenter(self, bbox):
        return int(bbox[1]+bbox[3])//2,int(bbox[0]+bbox[2])//2

    def getKeypoints2D(self, img, bbox):
        (startX, startY, endX, endY,_) = [int(i) for i in bbox]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[startY:endY,startX:endX]
        keypoints, descriptors = self.featureDetector.detectAndCompute(img,None)
        return keypoints, descriptors, img

    # pixels is a 3xN array
    def pixel2camera(self,pixels,z_estimate):
        cameraPoints3D = z_estimate*np.matmul(np.linalg.inv(self.cameraK),pixels)
        return cameraPoints3D

    def camera2pixel(self,cameraPoints3D):
        pixels = np.matmul(self.cameraK,(cameraPoints3D/cameraPoints3D[-1,:]))[:2]
        return pixels

    def featureMatch(self, descs1, descs2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs1,descs2, k=2)
        match = []
        dMatch = []
        for d1,d2 in matches:
            if d1.distance < 0.75*d2.distance:
                match.append(1)
                dMatch.append(d1)
            else:
                match.append(0)
        return np.array(match),dMatch

    # leave the keypoints as they are. Instead, divide the camera points by z_estd
    def globalKeypoint2camera(self, keypointMatches, objectModel, particleGlobal3D, T, R):
        Hcw = np.zeros((4,4))
        Hcw[:3,:3] = R
        Hcw[:3,3] = T
        Hcw[3,3] = 1
        points3D = np.zeros((4,np.sum(keypointMatches)))
        points3D[3,:] = np.ones((np.sum(keypointMatches)))
        p3Didx = 0
        for k,idx in enumerate(keypointMatches):
            if idx != 0:
                points3D[0:3,p3Didx] = objectModel.objectPoints[k]+particleGlobal3D
                p3Didx += 1
        cameraPoints3D = np.matmul(Hcw,points3D)
        cameraPoints3D = (cameraPoints3D/cameraPoints3D[-1,:])[:3,:]
        return cameraPoints3D
    
    def transformPoints(self, T, R, points3D):
        Hcw = np.zeros((4,4))
        Hcw[:3,:3] = R
        Hcw[:3,3] = T
        Hcw[3,3] = 1
        homogeneousPoints = np.vstack((points3D,np.ones(points3D.shape[1])))
        transformedPoints = np.matmul(Hcw,homogeneousPoints)
        transformedPoints = (transformedPoints/transformedPoints[-1,:])[:3,:]
        return transformedPoints


    def shiftKeypoints(self, kps, dmatch, bbox):
        pixels = []
        for idx in dmatch:
                pixels.append([kps[idx.trainIdx].pt[0]+bbox[0],kps[idx.trainIdx].pt[1]+bbox[1]])
        return np.array(pixels).T

    def scanObject(self, cvimg, bbox, xyz_array):
        keypoints, descriptors, cvimg = self.getKeypoints2D(cvimg, bbox)
        (startX, startY, endX, endY) = [int(i) for i in bbox]        
        campoints3D = xyz_array[startY:endY,startX:endX,:]
        origin3D = campoints3D[cvimg.shape[0]//2,cvimg.shape[1]//2,:]
        keypoints3D = Keypoints3D()
        for kp,desc in zip(keypoints,descriptors):
            # find a way to interpolate depth map
            u,v = int(kp.pt[1]), int(kp.pt[0])
            point3D = np.matmul(self.cameraR.T,campoints3D[u,v,:]-origin3D)
            if np.isnan(point3D).any() == False:
                keypoints3D.add(kp,desc,point3D)
        keypoints3D.numpyify()
        return keypoints3D,origin3D
