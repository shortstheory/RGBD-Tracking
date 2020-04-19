import numpy as np
class Keypoints3D:
    def __init__(self):
        self.keypoints = np.array([])
        self.descriptors = np.array([])
        self.objectPoints = np.array([])
    def add(self, kp, desc, objectPoint):
        np.append(self.keypoints, kp)
        np.append(self.descriptors, desc)
        np.append(self.objectPoints, objectPoint)

class BoundingBoxMessage:
    def __init__(self):
        self.bounding_boxes = []
        self.seq = 0
        self.timestamp = 0