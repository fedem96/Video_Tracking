import cv2
import os


class Face:
    def __init__(self, image, score):
        self.image = image
        self.score = score


class FaceDetector:
    def __init__(self, numFaces=15):
        # TODO implement max number of faces
        self.facesArchive = {}  # key=objectID; value=MinHeap of Faces (maximum numFaces for each objectID)
        self.nextID = 0
        self.numFaces = numFaces
        self.frontalface_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detectFaces(self, frame, objects_bboxes, objectsIDs):
        faces_bboxes = []
        for obj_bbox, objID in zip(objects_bboxes, objectsIDs):
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
        faces = []
        gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

        faces_bboxes = self.frontalface_cascade.detectMultiScale(gray, 1.3, 5)  # TODO capire questi parametri
        for (x, y, w, h) in faces_bboxes:
            imgFace = img_obj[y: y + h, x: x + w]
            imgFaceGray = gray[y: y + h, x: x + w]
            score = cv2.Laplacian(imgFaceGray, cv2.CV_64F).var()
            faces.append(Face(imgFace, score))

        return faces, faces_bboxes

    def dump(self, folder):

        for objID in self.facesArchive:
            objFolder = os.path.join(folder, "object_" + str(objID))
            if not os.path.exists(objFolder):
                os.makedirs(objFolder)

            count = 0
            for face in self.facesArchive[objID]:
                imgFile = os.path.join(objFolder, "face_" + str(count) + "-score_" + str(face.score) + ".png")
                cv2.imwrite(imgFile, face.image)
                count += 1


def main():
    # TODO define a test for this class
    pass


if __name__ == "__main__":
    main()
