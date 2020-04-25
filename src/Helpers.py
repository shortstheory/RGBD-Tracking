import numpy as np
class Keypoints3D:
    def __init__(self):
        self.keypoints = []
        self.descriptors = []
        self.objectPoints = []
    def add(self, kp, desc, objectPoint):
        self.keypoints.append(kp)
        self.descriptors.append(desc)
        self.objectPoints.append(objectPoint)
    def numpyify(self):
        self.keypoints   = np.array(self.keypoints)
        self.descriptors = np.array(self.descriptors)
        self.objectPoints = np.array(self.objectPoints)

class BoundingBoxMessage:
    def __init__(self):
        self.bounding_boxes = []
        self.seq = 0
        self.timestamp = 0