import copy

import cv2
import numpy as np


def fillHoles(img_bw):
    """
    Fill the holes (black) of the foreground (white) of a b/w image
    :param img_bw: b/w image to be processed
    :return: a copy of the given image, with filled holes
    """
    img_bw = copy.deepcopy(img_bw)
    contours = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
    for contour in contours:
        img_bw = cv2.fillPoly(img_bw, [contour], 255)
    return img_bw


def intersectionOverUnion(obj1, obj2):
    """
    Calculate the ratio between intersection area and union area of rectangles obj1 and obj2
    :param obj1: a rectangle in the form of (x,y,w,h)
    :param obj2: a rectangle in the form of (x,y,w,h)
    :return: intersection area over union area of specified rectangles
    """
    x1, y1, w1, h1 = obj1
    x2, y2, w2, h2 = obj2

    intersectionWidth = min(x1+w1, x2+w2) - max(x1, x2)
    intersectionHeight = min(y1+h1, y2+h2) - max(y1, y2)

    if intersectionWidth <= 0 or intersectionHeight <= 0:
        return 0.0

    intersectionArea = intersectionWidth * intersectionHeight
    unionArea = w1 * h1 + w2 * h2 - intersectionArea

    return intersectionArea / unionArea


def distance(obj1, obj2):
    """
    Calculate the L2 (Euclidean) distance between the centers of obj1 and obj2
    :param obj1: a rectangle in the form of (x,y,w,h)
    :param obj2: a rectangle in the form of (x,y,w,h)
    :return: L2 distance between centers of specified rectangles
    """
    x1, y1, w1, h1 = obj1
    x2, y2, w2, h2 = obj2

    cx1 = x1+w1//2
    cy1 = y1+h1//2

    cx2 = x2+w2//2
    cy2 = y2+h2//2

    return np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)


def draw_bboxes(image, bboxes, color, objIDs=None, scale=None, thickness=3):
    """
    Draw the bounding boxes (rectangles) on a copy of the image
    :param image: original image
    :param bboxes: list of bounding boxes, each one in the form of (x,y,w,h)
    :param color: color of the bounding boxes, in (b,g,r) format
    :param objIDs: list of IDs related to the bounding boxes, they will be drawn too (if None, only bboxes will be drawn)
    :param scale: scale factor to multiply all of (x,y,w,h)
    :param thickness: thickness of bounding boxes
    :return: a copy of the image, with bounding boxes applied over
    """
    imgCopy = copy.deepcopy(image)
    if scale is not None:
        bboxes = [[int(scale*x) for x in obj] for obj in bboxes]
    if objIDs is None:
        objIDs = [None for i in range(len(bboxes))]
    else:
        assert len(bboxes) == len(objIDs)
    for (x,y,w,h), objID in zip(bboxes, objIDs):
        cv2.rectangle(imgCopy, (x, y), (x + w, y + h), color, thickness)
        if objID is not None:
            cv2.putText(imgCopy, str(objID), (x+5, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return imgCopy


def concV(topImage, bottomImage):
    """
    Concatenate two images vertically
    :param topImage: image on top
    :param bottomImage: image on bottom
    :return: the image given by the vertical concatenation
    """
    if topImage is not None:
        return np.concatenate((topImage, bottomImage), axis=0)
    else:
        return bottomImage


def concH(leftImage, rightImage):
    """
    Concatenate two images horizontally
    :param leftImage: image on the left
    :param rightImage: image on the right
    :return: the image given by the horizontal concatenation
    """
    if leftImage is not None:
        return np.concatenate((leftImage, rightImage), axis=1)
    else:
        return rightImage


def mergeImgs(ll_images):
    """
    Merge a list of list of images
    :param ll_images: list of list of images, the inner list will be a row of images, so the outer list is a set of rows
    :return: the images given by vertical (outer list) and horizontal (inner list) concatenations
    """
    returnedImg = None

    for imgList in ll_images:
        imgRow = None
        for img in imgList:
            imgRow = concH(imgRow, img)
        returnedImg = concV(returnedImg, imgRow)

    return returnedImg



