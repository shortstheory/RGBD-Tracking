class BoundingBoxMessage:
    def __init__(self):
        self.bounding_boxes = []
        self.seq = 0
        self.timestamp = 0
        self.object_class = "UNKNOWN"