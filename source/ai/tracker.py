import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib
from utils.my_utils import linear_assignment, iou_batch, convert_bbox_to_z, convert_x_to_bbox, associate_detections_to_trackers
matplotlib.use('TkAgg')

# np.random.seed(0)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # History
        self.positions = []
        self.key_points = []
        self.faces = []
        self.faces_coordinates = []
        # self.faces_key_points = []
        self.bboxes = []
        self.poses_ypr = []
        self.poses_vn = []

    def print_(self):
        print("Person id: ", self.id, "pos len: ", len(self.positions), "kpts len: ", len(self.key_points), "faces len: ", len(self.faces))
        if len(self.positions) > 0:
            print("positions: ", self.positions)
        if len(self.key_points) > 0:
            print("each kpts len: ", len(self.key_points[0]))
        if len(self.faces) > 0:
            print("each face shape: ", type(self.faces[0]), self.faces[0].shape)
        if len(self.faces_coordinates) > 0:
            print("each face coords: ", self.faces_coordinates)
        # if len(self.faces_key_points) > 0:
        #     print("each face kpts: ", self.faces_key_points)

    def get_key_points(self):
        return self.key_points

    def get_bboxes(self):
        return self.bboxes

    def get_id(self):
        return self.id

    def get_faces(self):
        return self.faces

    def get_faces_coordinates(self):
        return self.faces_coordinates

    # def get_faces_key_points(self):
    #     return self.faces_key_points

    def get_poses_ypr(self):
        return self.poses_ypr

    def get_poses_vector_norm(self):
        return self.poses_vn

    def update_faces(self, face_image):
        self.faces.append(face_image)

    def update_faces_coordinates(self, face_image_coordinates):
        self.faces_coordinates.append(face_image_coordinates)

    # def update_faces_key_points(self, faces_key_points):
    #     """
    #
    #     Args:
    #         faces_key_points:
    #
    #     Returns:
    #
    #     """
    #     self.faces_key_points.append(faces_key_points)  # 0 nose, 1/2 left/right eye, 3/4 left/right ear

    def update_poses_ypr(self, poses_ypr):
        self.poses_ypr.append(poses_ypr)

    def update_poses_vector_norm(self, poses_vn):
        self.poses_vn.append(poses_vn)

    def update(self, bbox, kpts):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.positions.append([int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)])
        self.key_points.append(kpts)
        self.bboxes.append(list(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Sort(object):
    # max_age: Maximum number of frames to keep alive a track without associated detections.
    # min_hits: Minimum number of associated detections before track is initialised.
    # iou_threshold: Minimum IOU for match.
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # Each element represent a person
        self.frame_count = 0

    def print_(self):
        for i in self.trackers:
            print(i.print_())

    def get_trackers(self):
        aux = [a for a in self.trackers]
        return aux

    def update(self, dets, kpts):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        if len(dets) == 0:
            dets = (np.array([[0., 0., 0., 0., 0.]]))

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            # print("dets", dets, "m[0]", m[0], dets[m[0], :])
            self.trackers[m[1]].update(dets[m[0], :], kpts[m[0]])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
