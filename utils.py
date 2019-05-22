import copy
import os

import cv2
import numpy as np

frontalface_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#profileface_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def saveOutput(directory, detections):
    print("saving")
    for objID in detections:
        count = 0
        for objInfo in detections[objID]:
            print("object %d has %d face images " % (objID, len(objInfo[-1])))
            if len(objInfo[-1]) == 0:
                continue

            outputDir = os.path.join(directory, str(objID))
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            for image in objInfo[-1]:
                score = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                imgFile = os.path.join(outputDir, str(count) + "_" + str(score) + ".png")
                count += 1
                cv2.imwrite(imgFile, image)


def checkAndClearDir(directory):
    # creo cartelle o elimino contenuto
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        clearDirectory(directory)


# La funzione elimina i file all'interno della cartella che ha percorso = 'directorPath'
def clearDirectory(directoryPath):
    for file in os.listdir(directoryPath):
        file_path = os.path.join(directoryPath, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def findPrevious(obj, trackedObjects, previousIDs, iouThreshold=0.25):
    assert(len(trackedObjects) == len(previousIDs))
    maxIOU = 0
    id = None
    for (trkObj, trkObjID) in zip(trackedObjects, previousIDs):
        IOU = intersectionOverUnion(obj, trkObj)
        # IOU2 = get_iou(obj, trkObj)
        # assert IOU == IOU2
        if intersectionOverUnion(obj, trkObj) > maxIOU:
            maxIOU = IOU
            id = trkObjID
    if maxIOU >= iouThreshold:
        return id
    return None


#@deprecated
def removeAlreadyTracked(objects, trackedObjects, iouThreshold=0.2):
    newObjects = []
    for obj in objects:
        objAlreadyTracked = False
        for trkObj in trackedObjects:
            if intersectionOverUnion(obj, trkObj) > iouThreshold:
                objAlreadyTracked = True
                break
        if not objAlreadyTracked:
            newObjects.append(obj)
    return newObjects


def intersectionOverUnion(obj1, obj2):
    x1, y1, w1, h1 = obj1
    x2, y2, w2, h2 = obj2

    intersectionWidth = min(x1+w1, x2+w2) - max(x1, x2)
    intersectionHeight = min(y1+h1, y2+h2) - max(y1, y2)

    if intersectionWidth < 0 or intersectionHeight < 0:
        return 0.0

    intersectionArea = intersectionWidth * intersectionHeight
    unionArea = w1 * h1 + w2 * h2 - intersectionArea

    return intersectionArea / unionArea


# def get_iou(bb1, bb2):
#     assert bb1[0] < bb1[0]+bb1[2]
#     assert bb1[1] < bb1[1]+bb1[3]
#     assert bb2[0] < bb2[0]+bb2[2]
#     assert bb2[1] < bb2[1]+bb2[3]
#
#     # determine the coordinates of the intersection rectangle
#     x_left = max(bb1[0], bb2[0])
#     y_top = max(bb1[1], bb2[1])
#     x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
#     y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])
#
#     if x_right < x_left or y_bottom < y_top:
#         return 0.0
#
#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)
#
#     # compute the area of both AABBs
#     bb1_area = (bb1[0]+bb1[2] - bb1[0]) * (bb1[1]+bb1[3] - bb1[1])
#     bb2_area = (bb2[0]+bb2[2] - bb2[0]) * (bb2[1]+bb2[3] - bb2[1])
#
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
#     assert iou >= 0.0
#     assert iou <= 1.0
#     return iou

#@deprecated
def cleanAlreadyTracked(img, objects):
    imgCopy = copy.deepcopy(img)
    
    for (x, y, w, h) in objects:
        imgCopy[y:y+h, x:+x+w] = 0

    return imgCopy


def draw_bboxes(img, bboxes, color, objIDs=None, scale=None):
    imgCopy = copy.deepcopy(img)
    if scale is not None:
        bboxes = [[scale*x for x in obj] for obj in bboxes]
    if objIDs is None:
        objIDs = [None for i in range(len(bboxes))]
    for (x,y,w,h), objID in zip(bboxes, objIDs):
        cv2.rectangle(imgCopy, (x, y), (x + w, y + h), color, 3)
        if objID is not None:
            cv2.putText(imgCopy, str(objID), (x+5, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return imgCopy


def detectFaces(img, objects):
    faces = []
    parents = []
    obj_index = 0
    for (ox, oy, ow, oh) in objects:
        img_obj = img[oy:oy+oh, ox:+ox+ow]
        faces_found = detectFacesInObject(img_obj, (ox, oy))
        faces.extend(faces_found)
        parents.extend([obj_index for f in faces_found])
        obj_index += 1

    return faces, parents


def detectFacesInObject(img, obj_offset=(0,0)):
    (ox, oy) = obj_offset
    bboxes_faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = frontalface_cascade.detectMultiScale(gray, 1.3, 5)  # TODO capire questi parametri
    for (x, y, w, h) in faces:
        bboxes_faces.append([ox+x, oy+y, w, h])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)

        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return bboxes_faces


def flip(img):
    return np.flip(img, axis=1)


def concV(img1, img2):
    if img1 is not None:
        return np.concatenate((img1, img2), axis=0)
    else:
        return img2


def concH(img1, img2):
    if img1 is not None:
        return np.concatenate((img1, img2), axis=1)
    else:
        return img2


def mergeImgs(imgs):
    returnedImg = None

    for imgList in imgs:
        imgRow = None
        for img in imgList:
            imgRow = concH(imgRow, img)
        returnedImg = concV(returnedImg, imgRow)

    return returnedImg


def detectObjects_old(origImg, featImg):
    #featImg = cv2.cvtColor(featImg, cv2.COLOR_BGR2GRAY)
    imgCopy = copy.deepcopy(origImg)

    #featImg = cv2.dilate(featImg, kernel_dil, iterations=1)
    contours, hierarchy = cv2.findContours(featImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # hull = cv2.convexHull(contour)
        # hull_area = cv2.contourArea(hull)
        # solidity = float(area) / (hull_area + 0.00001)

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area
        #if area > 900 and extent > 0.2:  # bad    # questo criterio è molto importante
        if area > 4096 and hierarchy[0][pic][3] == -1:
            #x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(
                imgCopy, str(int(area)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1
            )

            faces = frontalface_cascade.detectMultiScale(featImg[y:y+h, x:x+w], 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(imgCopy, (x+fx, y+fy), (x+fx+fw, y+fy+fh), (255, 0, 0), 2)

    return imgCopy
