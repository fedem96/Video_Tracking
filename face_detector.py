import sys

import cv2
import os


class Face:
    def __init__(self, image, score):
        """
        Face constructor
        :param image: image of the face
        :param score: score of the face
        """
        self.image = image
        self.score = score


class FaceDetector:
    def __init__(self, maxFaces=15):
        """
        FaceDetector constructor
        :param maxFaces: maximum number of (best) faces the will be saved on the disk
        """
        self.facesArchive = {}  # key=objectID; value=list of faces (of class Face)
        self.nextID = 0
        self.maxFaces = maxFaces
        self.frontalface_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detectFaces(self, frame, objects_bboxes, objectsIDs, scale=None):
        """
        Detect faces inside the bounding boxes and accumulate them for later saving
        :param frame: complete frame containing all of the bounding boxes
        :param objects_bboxes: list of bounding boxes inside the frame
        :param objectsIDs: list of identifiers, related to the bounding boxes
        :param scale: scale factor to multiply all of (x,y,w,h)
        :return: a list of bounding boxes of the faces
        """
        faces_bboxes = []
        if scale is not None:
            objects_bboxes = [[int(scale*x) for x in obj] for obj in objects_bboxes]
        for obj_bbox, objID in zip(objects_bboxes, objectsIDs):
            if obj_bbox == [0,0,0,0]:
                sys.stderr.write("\nempty bounding box\n")
                continue
            (ox, oy, ow, oh) = obj_bbox
            img_obj = frame[oy:oy+oh, ox:+ox+ow]
            faces, faces_bb = self.detectFacesInObject(img_obj)

            if len(faces) == 0:
                continue

            if objID not in self.facesArchive:
                self.facesArchive[objID] = []

            self.facesArchive[objID].extend(faces)
            for face_bb in faces_bb:
                face_bb[0] += ox
                face_bb[1] += oy
            faces_bboxes.extend(faces_bb)

        return faces_bboxes

    def detectFacesInObject(self, img_obj):
        """
        Detect faces inside a single object
        :param img_obj: image of the object
        :return: a tuple (list of faces (class Face), list of bounding boxes)
        """
        faces = []
        gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

        faces_bboxes = self.frontalface_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces_bboxes:
            imgFace = img_obj[y: y + h, x: x + w]
            imgFaceGray = gray[y: y + h, x: x + w]
            score = cv2.Laplacian(imgFaceGray, cv2.CV_64F).var() * w**1.5
            faces.append(Face(imgFace, score))

        return faces, faces_bboxes

    def dump(self, folder):
        """
        Save faces to the specified folder
        :param folder: directory where to save the images of the faces
        """
        for objID in self.facesArchive:
            objFolder = os.path.join(folder, "object_" + str(objID))
            if not os.path.exists(objFolder):
                os.makedirs(objFolder)

            count = 0
            faces = sorted(self.facesArchive[objID], key=lambda face: face.score, reverse=True)
            for face in faces:
                imgFile = os.path.join(objFolder, "face_%02d" % count + "-score_" + str(int(face.score//1000)) + ".png")
                cv2.imwrite(imgFile, face.image)
                count += 1
                if count >= self.maxFaces:
                    break


def main():
    # TODO define a test for this class
    pass


if __name__ == "__main__":
    main()
