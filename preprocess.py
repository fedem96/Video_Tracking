import cv2
import numpy as np


class ProcessPipeline:

    def __init__(self):
        """
        ProcessPipeline constructor
        """
        self.functions = []
        self.params = []
        self.intermediateOutputs = []
        self.intermediateOutputsBGR = []

    def add(self, function, **kwargs):
        """
        Enqueue one function in the pipeline
        :param function: a function that takes a b/w image, process it and return the processed b/w image
        :param kwargs: kwargs for the function
        :return: reference of self
        """
        self.functions.append(function)
        self.params.append(kwargs)
        return self

    def clear(self):
        """
        Remove all functions from the pipeline
        :return: reference of self
        """
        self.functions = []
        self.params = []
        self.intermediateOutputs = []
        return self

    def process(self, fgmask):
        """
        Execute all the functions in the pipeline
        :param fgmask: b/w image to be processed
        :return: b/w image processed by all the functions in the pipeline
        """
        self.intermediateOutputs = [fgmask]
        self.intermediateOutputsBGR = [cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)]
        for function, kwargs in zip(self.functions, self.params):
            fgmask = function(fgmask, **kwargs)
            self.intermediateOutputs.append(fgmask)
            self.intermediateOutputsBGR.append(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))

        return self.intermediateOutputs[-1]


class CompositeBackgroundSubtractor:
    def __init__(self, *args):
        """
        CompositeBackgroundSubtractor constructor
        :param args: two or more background subtractors
        """
        self.bgSubtractors = args

    def apply(self, frame):
        """
        Background subtraction is executed with each background subtractor, result is the OR of the subtractions
        :param frame: image to segment
        :return: OR between results of background subtractions
        """
        fgmaskTot = np.zeros((frame.shape[0],frame.shape[1]), dtype="uint8")
        for bgSub in self.bgSubtractors:
            fgmask = bgSub.apply(frame)
            fgmask[fgmask != 255] = 0
            fgmaskTot = np.maximum(fgmask, fgmaskTot)
        return fgmaskTot

