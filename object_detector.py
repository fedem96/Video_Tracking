import copy

import cv2


class ObjectDetector:

    def __init__(self, bgSubtractor, processPipeline):
        self.fgbg = bgSubtractor
        self.pipeline = copy.deepcopy(processPipeline)

    def detect(self, frame, minArea=5000):
        fgmask = self.fgbg.apply(frame)
        fgmask[fgmask != 255] = 0
        img_bw = self.pipeline.process(fgmask)

        objects = []
        contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # hull = cv2.convexHull(contour)
            # hull_area = cv2.contourArea(hull)
            # solidity = float(area) / (hull_area + 0.00001)

            x, y, w, h = cv2.boundingRect(contour)
            #rect_area = w * h
            #extent = float(area) / rect_area
            # if area > 900 and extent > 0.2:  # bad    # questo criterio è molto importante
            if area >= minArea and hierarchy[0][pic][3] == -1:
                objects.append((x, y, w, h))

        # controllare che non ci siano bbox dentro altri bbox

        return objects
