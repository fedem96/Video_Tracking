import copy

import cv2


class ObjectDetector:

    def __init__(self, bgSubtractor, processPipeline):
        """
        ObjectDetector constructor
        :param bgSubtractor: a background subtractor algorithm (cv2.BackgroundSubtractor)
        :param processPipeline: a ProcessPipeline objects, which specifies processing steps to apply after the background subtraction and before bounding boxes creation
        """
        self.bgSubtractor = bgSubtractor
        self.pipeline = copy.deepcopy(processPipeline)

    def detect(self, frame, minArea=0.1, maxArea=0.5):
        """
        Detect objects on a given frame (without recognizing them)
        :param frame: image where to search for objects
        :param minArea: minimum area of contour bounding rect to consider it an object (minArea is intended as the ratio w.r.t. the frame area)
        :param maxArea: maximum area of contour bounding rect to consider it an object (maximum is intended as the ratio w.r.t. the frame area)
        :return: a list of bounding boxes, each one in the form of (x,y,w,h)
        """
        frameArea = frame.shape[0] * frame.shape[1]
        fgmask = self.bgSubtractor.apply(frame)     # apply background subtractor
        fgmask[fgmask != 255] = 0   # remove grays
        fgmask = self.pipeline.process(fgmask)  # apply pipeline processing steps

        objects = []
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w*h
            # if area is in the desired interval and the contour has no parent
            if minArea * frameArea <= area <= maxArea * frameArea and hierarchy[0][i][3] == -1:
                objects.append((x, y, w, h))

        return objects
