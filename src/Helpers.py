class Keypoint3D:
    def __init__(self,kp,desc,point3D):
        self.kp = kp
        self.desc = desc
        self.point3D = point3D

class BoundingBoxMessage:
    def __init__(self):
        self.bounding_boxes = []
        self.seq = 0
        self.timestamp = 0