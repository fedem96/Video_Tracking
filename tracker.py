from utils import *


class Tracker:

    nextID = 0

    def __init__(self, tracker, id=None, eps=5):
        """
        Tracker constructor
        :param tracker: object of type cv2.Tracker
        :param id: id of the tracker; if None, it will be auto-assigned
        :param eps: difference in pixels within which it is considered that the object's bounding box is not moving
        """
        self.position = None
        self.speed = None
        self.tracker = tracker
        self.numFailures = 0
        self.eps = eps
        self.lastBBox = None
        self.lastSuccess = None
        if id is None:
            self.id = Tracker.nextID
            Tracker.nextID += 1
        else:
            self.id = id

    def init(self, frame, obj_bbox):
        """
        Tracker initialization
        :param frame: first frame seen from the tracker
        :param obj_bbox: bounding box (as (x,y,w,h)) of the object to be tracked
        :return: True if initialization went successfully, False otherwise
        """
        self.position = obj_bbox[0] + obj_bbox[2] // 2, obj_bbox[1] + obj_bbox[3] // 2  # x+w//2, y+h//2
        return self.tracker.init(frame, obj_bbox)

    def update(self, frame):
        """
        Update the tracker, finding the new most likely bounding box for the target
        :param frame: the frame where to search for the object
        :return: a tuple (s, b); s is a boolean that indicates if target has been successfully located; b is bounding box that represent the new target location, if s=True was returned
        """
        s, b = self.tracker.update(frame)
        b = [int(max([k, 0])) for k in b]
        b[0] = min(b[0], frame.shape[1]-1)           # x < frame_width
        b[1] = min(b[1], frame.shape[0]-1)           # y < frame_height
        b[2] = min(b[2], frame.shape[1]-1-b[0])      # w < frame_width - x
        b[3] = min(b[3], frame.shape[0]-1-b[1])      # h < frame_height - y
        position = b[0] + b[2]//2, b[1] + b[3]//2    # x+w//2, y+h//2
        self.speed = (position[0]-self.position[0], position[1]-self.position[1])
        if not s:
            self.numFailures += 1
        elif abs(self.speed[0]) <= self.eps and abs(self.speed[1]) <= self.eps:
            self.numFailures += 2
        else:
            self.numFailures = 0
        self.position = position
        return s, b


class TrackerManager:
    def __init__(self, nameDefaultTracker, maxFailures=80):
        """
        TrackerManager constructor
        :param nameDefaultTracker: name of the tracker that will be created in addTracker (if not specified otherwise there)
        :param maxFailures: maximum number of consecutive frames in which the tracker can fail, beyond which it will be automatically destroyed
        """
        self.trackers = []
        self.nameDefaultTracker = nameDefaultTracker
        self.maxFailures = maxFailures

    def addTracker(self, frame, obj_bbox, trackerName=None):
        """
        Add a tracker to the list of managed trackers, initializing it with its first frame and bbox
        :param frame: first frame for the created tracker
        :param obj_bbox: bounding box (as (x,y,w,h)) of the object to be tracked
        :param trackerName: name of the tracker that will be created in addTracker (if None, default is considered)
        :return: the created tracker
        """
        if trackerName is None:
            trackerName = self.nameDefaultTracker
        if trackerName == "MOSSE":
            tracker = Tracker(cv2.TrackerMOSSE_create())
        elif trackerName == "KCF":
            tracker = Tracker(cv2.TrackerKCF_create())
        elif trackerName == "CSRT":
            tracker = Tracker(cv2.TrackerCSRT_create())
        else:
            print("unknown tracker")
            exit(1)

        tracker.init(frame, obj_bbox)
        self.trackers.append(tracker)
        return tracker

    def _update(self, frame):
        """
        Updates all trackers on the given frame. No merge with detection is considered here.
        :param frame: the frame where to search for the objects
        :return: a tuple of lists (ls, lb); ls is a list of boolean (True if the object is successfully located); lb is a list of bounding boxes, each of them represents an object's location
        """
        successes = []
        bboxes = []
        for i, tracker in enumerate(self.trackers):
            s, b = tracker.update(frame)
            if not s and b == [0,0,0,0] and tracker.lastBBox is not None:
                b = tracker.lastBBox
            else:
                tracker.lastBBox = b
            tracker.lastSuccess = s
            successes.append(s)
            bboxes.append(b)
            
        idxsSuppressed = self.suppressDuplicateTrackers(bboxes, frame.shape)
        successes = [successes[i] for i in range(len(successes)) if i not in idxsSuppressed]
        bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in idxsSuppressed]
        
        return successes, bboxes

    def update(self, frame, detectedObjects=None, maintainDetected=True):
        """
        Updates all trackers on the given frame. Eventually merge with detection.
        :param frame: the frame where to search for the objects
        :param detectedObjects: the list of bounding boxes (as (x,y,w,h)) given by an external object detector
        :param maintainDetected: True to maintain detector's bounding boxes in case of overlaps with trackers' bounding boxes, False to maintain the latter
        :return: a tuple of lists (ls, lb); ls is a list of boolean (True if the object is successfully located); lb is a list of bounding boxes, each of them represents an object's location
        """
        successes, bboxes = self._update(frame)  # update all trackers (without merging bounding boxes)

        if detectedObjects is not None and detectedObjects != []:
            trkIDs = self.getIDs()   # get the IDs of tracked objects
            successes, bboxes, objIDs, changes = self.mergeBBoxes(successes, bboxes, detectedObjects, trkIDs=trkIDs, maintainDetected=maintainDetected)
            # these 4 lists above are all of the same length, each one is related to the others (i.e., the same index refers to the same object)
            # successes: list of booleans; element i-th is True if tracker of index (not id!) i has successfully located the target
            # bboxes:    list of bounding boxes; element i-th is the bounding box (x,y,w,h) of object i-th
            # objIDs:    list of identifiers (integers); element i-th is >= 0 if the object already had an identifier, otherwise -1
            # changes:   list of booleans; element i-th is True if bounding box of index (not id!) i have been detected or changed by the detector (in this frame)
            for bbox, objID, change in zip(bboxes, objIDs, changes):
                if change:
                    if objID == -1:
                        self.addTracker(frame, bbox)
                    else:
                        self.reinitTracker(objID, frame, bbox)
        return successes, bboxes

    def mergeBBoxes(self, trkSuccesses, trackedObjects, detectedObjects, threshold=0.2, trkIDs=None, maintainDetected=True):
        """
        Merge trackers' and detector's bounding boxes, resolving the conflicts (overlaps)
        :param trkSuccesses: successes returned by _update(frame)
        :param trackedObjects: bboxes returned by _update(frame)
        :param detectedObjects: the list of bounding boxes (as (x,y,w,h)) given by an external object detector
        :param threshold: minimum intersection over union to consider overlap between to different bounding boxes
        :param trkIDs: identifiers of tracked objects
        :param maintainDetected: True to maintain detector's bounding boxes in case of overlaps with trackers' bounding boxes, False to maintain the latter
        :return: a 4-tuple of lists (ls, lb, ); ls is a list of boolean (True if the object is successfully located); lb is a list of bounding boxes, each of them represents an object's location
        """
        assert trkIDs is not None
        assert len(trkIDs) == len(trackedObjects)
        assert len(trkSuccesses) == len(trackedObjects)
        if detectedObjects is None:
            detectedObjects = []

        bboxes = copy.deepcopy(detectedObjects)
        successes = [True for x in bboxes]
        objIDs = [-1 for x in bboxes]
        changes = [True for x in bboxes]

        toAdd = []
        for t, (trkObj, trkID) in enumerate(zip(trackedObjects, trkIDs)):
            iouMax = 0
            iMax = -1
            for d in range(len(detectedObjects)):
                detObj = detectedObjects[d]
                tmp_iou = intersectionOverUnion(detObj, trkObj)
                if tmp_iou >= threshold and tmp_iou > iouMax:
                    iouMax = tmp_iou
                    iMax = d
            if iMax != -1:
                objIDs[iMax] = trkID
                if not maintainDetected:
                    bboxes[iMax] = trkObj
                    changes[iMax] = False
                    successes[iMax] = trkSuccesses[t]
            else:
                toAdd.append(t)

        for t in toAdd:
            bboxes.append(trackedObjects[t])
            objIDs.append(trkIDs[t])
            changes.append(False)
            successes.append(trkSuccesses[t])

        successes, bboxes, objIDs, changes = [list(l) for l in zip(*sorted(zip(successes, bboxes, objIDs, changes), key=lambda x: x[2] + 10**8*(1-np.sign(x[2]))*abs(x[2])))]

        return successes, bboxes, objIDs, changes

    def removeDeadTrackers(self):
        """
        Remove all trackers that has exceeded the number of maximum allowed failures
        """
        self.trackers = [tracker for tracker in self.trackers if tracker.numFailures <= self.maxFailures]

    def getIDs(self):
        """
        Obtain the identifiers of the managed trackers
        :return: a list of identifiers of the managed trackers
        """
        return [tracker.id for tracker in self.trackers]

    def removeTracker(self, objID):
        """
        Remove a specific tracker, identified by its own ID
        :param objID: identifier of the tracker to be removed
        :return: True if a tracker is removed, False otherwise
        """
        for t in range(len(self.trackers)):
            if self.trackers[t].id == objID:
                self.trackers.pop(t)
                return True
        return False

    def reinitTracker(self, objID, frame, obj_bbox):
        """
        Create and initialize a new tracker that replaces an existing one, while maintaining the same identifier (this is necessary when we want to force the change of object bounding box)
        :param objID:
        :param frame:
        :param obj_bbox:
        :return:
        """
        for t in range(len(self.trackers)):
            if self.trackers[t].id == objID:
                clsName = str(self.trackers[t].tracker.__class__)
                clsName = clsName[clsName.index("'")+1: clsName.rindex("'")]
                tracker = Tracker(eval(clsName + "_create()"), id=objID)
                tracker.init(frame, obj_bbox)
                self.trackers[t] = tracker
        return False

    def suppressDuplicateTrackers(self, bboxes, frame_shape, threshold=0.75):
        """
        Suppress different trackers that are tracking the same object, leaving one tracker only for object. For the similarity score, are considered the intersection over union, the distance between centers, and the difference in speed
        :param bboxes: list of bounding boxes of the objects
        :param frame_shape: shape of the frame, useful to normalize the distance between bounding boxes
        :param threshold: minimum value of similarity score between two different bounding boxes to consider them referring to the same object
        :return: list of indexes (not identifiers) of removed trackers
        """
        removingIndexes = []
        affinityList = []
        for i, (trackerI, bboxI) in enumerate(zip(self.trackers, bboxes)):
            for j, (trackerJ, bboxJ) in enumerate(zip(self.trackers, bboxes)):
                if i == j or bboxI[0]*bboxI[1] < bboxJ[0]*bboxJ[1]:
                    continue
                iou = intersectionOverUnion(bboxI, bboxJ)
                dist = distance(bboxI, bboxJ)
                normDist = dist / np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
                deltaSpeed = np.sqrt((trackerI.speed[0]-trackerJ.speed[0])**2 + (trackerI.speed[1]-trackerJ.speed[1])**2) /\
                             np.sqrt((2*frame_shape[0])**2 + (2*frame_shape[1])**2)
                score = (iou + (1-normDist) + (1-deltaSpeed)) / 3    # number between 0 and 1: if high, bboxI and bboxJ refers to the same object
                if score >= threshold:
                    affinityList.append([i, j, score])
        affinityList = sorted(affinityList, key=lambda x:x[2], reverse=True)
        for i, j, score in affinityList:
            removingIndexes.append(j)
        removingIndexes = sorted(removingIndexes, reverse=True)
        for i, j in enumerate(removingIndexes):
            if i > 0 and removingIndexes[i-1] == j:
                continue
            self.trackers.pop(j)
        return removingIndexes
