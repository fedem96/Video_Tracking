import cv2
import os
import imutils
from timeit import default_timer as timer

from object_detector import ObjectDetector
from face_detector import FaceDetector
from preprocess import ProcessPipeline, CompositeBackgroundSubtractor
from tracker import TrackerManager
from utils import fillHoles, draw_bboxes


def main():
    ''' input '''
    # file video
    # captureSource = 'video/living_room_clear_allWhite_light.mp4' # questo per vedere quando bg subtractor non funzionano
    #captureSource = "video/video_116.mp4"
    captureSource = "video/video_205.mp4"
    #captureSource = "video/video_white.mp4"
    # webcam
    #captureSource = 0
    cap = cv2.VideoCapture(captureSource)

    ''' trackers typology '''
    trackerName = "CSRT"  # "MOSSE" | "KCF" | "CSRT"
    tm = TrackerManager(trackerName, maxFailures=20)

    ''' parameters '''
    period = 1                  # length of the period: only on the first frame of the period we detect objects (instead, we track them in every frame)
    maintainDetected = True     # True if in transition frames, in case of overlapping bboxes,  we want to keep those of the detector (False if we want to keep those of the tracker)
    frameWidth = 512

    ''' some background subtractor with default params '''
    # bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    # bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,	detectShadows=True)
    # bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
    # bgSubtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
    # bgSubtractor = CompositeBackgroundSubtractor(bgSubtractor1, bgSubtractor2, ...)

    ''' background subtractor '''
    #bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)     # good for video/video_116.mp4
    bgSubtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500.0, detectShadows=True) # good for video/video_205.mp4
    # bgSubtractor = CompositeBackgroundSubtractor(
    #     cv2.bgsegm.createBackgroundSubtractorMOG(history=600, nmixtures=3, backgroundRatio=0.2, noiseSigma=2.3),
    #     cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=14, detectShadows=True)) # good for video/video_white.mp4

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pipeline = ProcessPipeline()
    pipeline \
        .add(cv2.medianBlur, ksize=5) \
        .add(cv2.dilate, kernel=kernel) \
        .add(cv2.dilate, kernel=kernel) \
        .add(cv2.dilate, kernel=kernel) \
        .add(cv2.dilate, kernel=kernel) \
        .add(cv2.dilate, kernel=kernel) \
        .add(fillHoles) \
        .add(cv2.erode, kernel=kernel) \
        .add(cv2.erode, kernel=kernel) \

    od = ObjectDetector(bgSubtractor, pipeline)
    fd = FaceDetector()

    ''' auto-definition of output folder '''
    outputDir = "output"
    if captureSource == 0:
        outputDir = os.path.join(outputDir, "webcam")
    else:
        outputDir = os.path.join(outputDir, captureSource[:captureSource.find(".")])
    outputDir = os.path.join(outputDir, trackerName)
    print("Tracking video '%s' with tracker %s" % (captureSource, trackerName))

    ''' cycle begins '''
    frameNumber = 0
    frames = 0
    seconds = 0
    eta = 0.05
    totalTime = 0
    show = True
    oneSkipOnly = False
    while True:

        ''' handle input: esc to quit; space to pause/start; "n" to go one frame at a time '''
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord(' ') or oneSkipOnly:
            show = not show
            oneSkipOnly = False
        elif k == ord('n'):
            show = True
            oneSkipOnly = True
        if not show:
            if oneSkipOnly:
                show = False
            continue

        start = timer()
        ''' reading next frame '''
        ret, frameOrig = cap.read()
        if not ret:
            break
        frameOrig = cv2.flip(frameOrig, 1)
        #frame = imutils.resize(frameOrig, width=frameOrig.shape[1]//scale)
        frame = imutils.resize(frameOrig, width=frameWidth)
        scale = frameOrig.shape[1] / frameWidth

        detectedObjects = []
        if frameNumber % period == 0:
            ''' detection by background subtraction '''
            detectedObjects = od.detect(frame)
            ''' objects tracking, faces detection'''

        ''' tracking '''
        success, objects = tm.update(frame, detectedObjects, maintainDetected=maintainDetected)
        objIDs = tm.getIDs()
        tm.removeDeadTrackers()

        failed_objects = [obj for suc, obj in zip(success, objects) if not suc]
        failed_objIDs = [objID for suc, objID in zip(success, objIDs) if not suc]
        succ_objIDs = [objID for suc, objID in zip(success, objIDs) if suc]
        objects = [obj for suc, obj in zip(success, objects) if suc]

        assert len(objects) == len(succ_objIDs)
        assert len(failed_objIDs) == len(failed_objects)

        faces_bboxes = fd.detectFaces(frameOrig, objects, objIDs, scale=scale)

        ''' images merging and show '''
        frameOrig = draw_bboxes(frameOrig, objects, (255,0,0), succ_objIDs, scale=scale)
        frameOrig = draw_bboxes(frameOrig, failed_objects, (0,0,255), failed_objIDs, scale=scale)
        frameOrig = draw_bboxes(frameOrig, faces_bboxes, (0,255,0))
        frameOrig = cv2.resize(frameOrig, (640, 640))
        cv2.imshow('frame', frameOrig)

        frameNumber += 1
        end = timer()
        frames = eta + (1-eta)*frames
        seconds = eta * (end-start) + (1-eta)*seconds
        print("\rFrame: %04d    FPS: %03d   Active trackers: %02d    Failed trackers: %02d           " %
              (frameNumber, int(frames // seconds), len(objects), len(failed_objects)), end="")
        totalTime += end - start

    cap.release()
    cv2.destroyAllWindows()

    fd.dump(outputDir)

    avgFPS = str(round(frameNumber / totalTime, 2))
    print("\rAverage FPS: " + avgFPS)
    with open(os.path.join(outputDir, "info.txt"), "w") as file:
        bgSubClsName = str(bgSubtractor.__class__)
        bgSubClsName = bgSubClsName[bgSubClsName.index("'") + 1: bgSubClsName.rindex("'")]
        file.write("tracker: " + trackerName + "\n")
        file.write("background subtractor: " + bgSubClsName + "\n")
        file.write("average FPS: " + avgFPS + "\n")


if __name__ == "__main__":
    main()
