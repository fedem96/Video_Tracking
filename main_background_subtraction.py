import cv2
import imutils

from object_detector import ObjectDetector
from preprocess import CompositeBackgroundSubtractor, ProcessPipeline
from utils import mergeImgs, draw_bboxes, fillHoles

"""
For the same input stream, are compared simultaneously different background subtractors and the corresponding results in terms of object detection.
After subtraction of the background and before the actual detection, functions can be applied to the b/w mask returned by the background subtractor, in order to have a better detection.
"""


def main():
    ''' input '''
    # choose the input stream
    #captureSource = 0  # webcam
    #captureSource = 'video/video_116.mp4'
    captureSource = 'video/video_205.mp4'
    #captureSource = 'video/video_white.mp4'
    cap = cv2.VideoCapture(captureSource)

    ''' some background subtractor with default params '''
    # bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    # bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,	detectShadows=True)
    # bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
    # bgSubtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
    # bgSubtractor = CompositeBackgroundSubtractor(bgSubtractor1, bgSubtractor2, ...)

    ''' list of background subtractors '''
    backgroundSubtractors = [
        cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True),     # good for video/video_116.mp4
        cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500.0, detectShadows=True), # good for video/video_205.mp4
        CompositeBackgroundSubtractor(
                 cv2.bgsegm.createBackgroundSubtractorMOG(history=600, nmixtures=3, backgroundRatio=0.2, noiseSigma=2.3),
                 cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=14, detectShadows=True)) # good for video/video_white.mp4
    ]

    ''' pipeline steps applied after background subtraction '''
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

    ''' detectors creation and beginning of video analysis '''
    detectors = [ObjectDetector(bgSub, pipeline) for bgSub in backgroundSubtractors]
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

        ''' reading next frame '''
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=512)
        frame = cv2.flip(frame, 1)

        ''' detection '''
        color = (255,0,0)   # blue
        outputFrames = [
            draw_bboxes(frame, detector.detect(frame), color) for detector in detectors
        ]

        ''' show frames '''
        imgOutput = mergeImgs([
            [*detector.pipeline.intermediateOutputsBGR, outputFrame] for detector, outputFrame in zip(detectors, outputFrames)
        ])
        imgOutput = cv2.resize(imgOutput, (1920, 1000))
        cv2.imshow('frame', imgOutput)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
