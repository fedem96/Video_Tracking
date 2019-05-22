import cv2
from preprocess import ProcessPipeline
import imutils

from utils import *
from object_detector import ObjectDetector
from face_detector import FaceDetector


class Tracker:

    def __init__(self, tracker, id):
        self.tracker = tracker
        self.id = id

    def getID(self):
        return self.id

    def __getattr__(self, *args):
        return self.tracker.__getattribute__(*args)


class TrackerManager:
    def __init__(self):
        self.trackers = cv2.MultiTracker_create()
        self.objectsIDs = []
        self.nextID = 0
        # self.pastObjects = []

    def addTracker(self, trackerName, frame, obj, objID):
        if trackerName == "MOSSE":
            tracker = cv2.TrackerMOSSE_create()
        elif trackerName == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif trackerName == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            print("unknown tracker")
            exit(1)
        self.trackers.add(tracker, frame, obj)
        self.objectsIDs.append(objID)

    def deleteAllTrackers(self):
        self.trackers = cv2.MultiTracker_create()
        self.objectsIDs = []

    def update(self, frame, detectedObjects):
        (success, trackedObjects) = self.trackers.update(frame)
        trackedObjects = [[int(min([max([k, 0]), 1920])) for k in obj] for obj in trackedObjects] # TODO remove that 1920

        affinityList = []
        previousObjectsIDs = self.objectsIDs
        self.objectsIDs = []
        for d in range(len(detectedObjects)):
            detObj = detectedObjects[d]
            self.objectsIDs.append(-1)
            for t in range(len(trackedObjects)):
                trkObj = trackedObjects[t]
                iou = intersectionOverUnion(detObj, trkObj)
                affinityList.append([d, t, iou])
        # sort by decreasing Intersection over Union
        affinityList = sorted(affinityList, key=lambda x:x[2], reverse=True)
        for d, t, iou in affinityList:
            if previousObjectsIDs[t] in self.objectsIDs or self.objectsIDs[d] != -1:
                continue
            self.objectsIDs[d] = previousObjectsIDs[t]

        for d in range(len(detectedObjects)):
            if self.objectsIDs[d] == -1:
                self.objectsIDs[d] = self.nextID
                self.nextID += 1

        return self.objectsIDs


def main():
    ''' input '''
    # file video
    #captureSource = 'video/video.mp4'
    #captureSource = 'video/video2.mp4'
    #captureSource = 'video/video3.mp4'
    captureSource = 'video/video_116_1.mp4'
    #captureSource = 'video/prova.mp4'
    #captureSource = 'video/far.mp4'
    # webcam
    #captureSource = 0
    cap = cv2.VideoCapture(captureSource)

    ''' trackers typology '''
    trackerName = "MOSSE"  # "MOSSE" | "KCF" | "CSRT"
    tm = TrackerManager()

    ''' detector definition '''
    #bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    #bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,	detectShadows=True)
    #bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8) #orig, bad
    #bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=20, decisionThreshold=0.9999) # a little better
    bgSubtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

    pipeline = ProcessPipeline()
    pipeline \
        .add(cv2.medianBlur, ksize=5) \
        .add(cv2.dilate, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    # fgmask_f = cv2.filter2D(fgmask, -1, kernel)
    od = ObjectDetector(bgSubtractor, pipeline)
    fd = FaceDetector()

    ''' auto-definition of output folder '''
    outputDir = "output"
    if captureSource == 0:
        outputDir = os.path.join(outputDir, "webcam")
    else:
        outputDir = os.path.join(outputDir, captureSource[:captureSource.find(".")])
    outputDir = os.path.join(outputDir, trackerName)

    ''' cycle begins '''
    show = True
    scale = 2
    while True:

        ''' handle input '''
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord(' '):
            show = not show
        if not show:
            continue

        ''' reading next frame '''
        ret, frameOrig = cap.read()
        if not ret:
            break
        frameOrig = cv2.flip(frameOrig, 1)
        frame = imutils.resize(frameOrig, width=frameOrig.shape[1]//scale)

        ''' detection by background subtraction '''
        objects = od.detect(frame)

        ''' objects tracking, faces detection'''
        objIDs = tm.update(frame, objects)
        faces_bboxes = fd.detectFaces(frame, objects, objIDs)
        tm.deleteAllTrackers()
        for obj, objID in zip(objects, objIDs):
            tm.addTracker(trackerName, frame, obj, objID)

        ''' images merging and show '''
        frameOrig = draw_bboxes(frameOrig, objects, (255,0,0), objIDs, scale=scale)
        frameOrig = draw_bboxes(frameOrig, faces_bboxes, (0,255,0), scale=scale)
        frameOrig = cv2.resize(frameOrig, (640, 640))
        cv2.imshow('frame', frameOrig)

    cap.release()
    cv2.destroyAllWindows()

    fd.dump(outputDir)


if __name__ == "__main__":
    main()
