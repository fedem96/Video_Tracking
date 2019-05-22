import cv2
import imutils

from object_detector import ObjectDetector
from utils import mergeImgs, draw_bboxes


class ProcessPipeline:

    def __init__(self):
        self.functions = []
        self.params = []
        self.intermediateOutputs = []
        self.intermediateOutputsBGR = []

    def add(self, function, **kwargs):
        self.functions.append(function)
        self.params.append(kwargs)
        return self

    def clear(self):
        self.functions = []
        self.params = []
        self.intermediateOutputs = []

    def process(self, img_bw):
        self.intermediateOutputs = [img_bw]
        self.intermediateOutputsBGR = [cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)]
        for function, kwargs in zip(self.functions, self.params):
            img_bw = function(img_bw, **kwargs)
            self.intermediateOutputs.append(img_bw)
            self.intermediateOutputsBGR.append(cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR))

        return self.intermediateOutputs[-1]


def main():
    #cap = cv2.VideoCapture('video/video3.mp4')
    cap = cv2.VideoCapture('video/video_116_1.mp4')
    fgbgMOG = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,	detectShadows=True)
    fgbgGMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=20, decisionThreshold=0.9999)
    fgbgKNN = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    pipeline = ProcessPipeline()
    pipeline \
        .add(cv2.medianBlur, ksize=5) \
        .add(cv2.dilate, kernel=kernel) \
        #.addFunction(cv2.medianBlur, ksize=3)
    detectorMOG = ObjectDetector(fgbgMOG, pipeline)
    detectorMOG2 = ObjectDetector(fgbgMOG2, pipeline)
    detectorGMG = ObjectDetector(fgbgGMG, pipeline)
    detectorKNN = ObjectDetector(fgbgKNN, pipeline)

    show = True
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
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=frame.shape[1] // 2)
        frame = cv2.flip(frame, 1)

        ''' detection '''
        color = (255,0,0)
        frameMOG = draw_bboxes(frame, detectorMOG.detect(frame), color)
        frameMOG2 = draw_bboxes(frame, detectorMOG2.detect(frame), color)
        frameGMG = draw_bboxes(frame, detectorGMG.detect(frame), color)
        frameKNN = draw_bboxes(frame, detectorKNN.detect(frame), color)

        ''' show frames '''
        imgOutput = mergeImgs([
            [*detectorMOG.pipeline.intermediateOutputsBGR, frameMOG],
            [*detectorMOG2.pipeline.intermediateOutputsBGR, frameMOG2],
            [*detectorGMG.pipeline.intermediateOutputsBGR, frameGMG],
            [*detectorKNN.pipeline.intermediateOutputsBGR, frameKNN]
        ])
        imgOutput = cv2.resize(imgOutput, (1920, 1000))
        cv2.imshow('frame', imgOutput)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
