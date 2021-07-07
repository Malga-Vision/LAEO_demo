import numpy as np
from scipy.spatial import distance as dist
from utils.labels import pose_id_part, pose_id_part_openpose, rev_pose_id_part_openpose, rev_pose_id_part
import cv2
import os
import json


def rescale_bb(boxes, pad, im_width, im_height):
    """
    Modify in place the bounding box coordinates (percentage) to the new image width and height

    Args:
        :boxes (numpy.ndarray): Array of bounding box coordinates expressed in percentage [y_min, x_min, y_max, x_max]
        :pad (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                        the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                        the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                        the coordinates changes)
        :im_width (int): The new image width
        :im_height (int): The new image height

    Returns:
    """

    right_padding = pad[0]
    bottom_padding = pad[1]

    if bottom_padding != 0:
        for box in boxes:
            y_min, y_max = box[0] * im_height, box[2] * im_height  # to pixels
            box[0], box[2] = y_min / (im_height - pad[1]), y_max / (im_height - pad[1])  # back to percentage

    if right_padding != 0:
        for box in boxes:
            x_min, x_max = box[1] * im_width, box[3] * im_width  # to pixels
            box[1], box[3] = x_min / (im_width - pad[0]), x_max / (im_width - pad[0])  # back to percentage


def rescale_key_points(key_points, pad, im_width, im_height):
    """
    Modify in place the bounding box coordinates (percentage) to the new image width and height

    Args:
        :key_points (numpy.ndarray): Array of bounding box coordinates expressed in percentage [y_min, x_min, y_max, x_max]
        :pad (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                        the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                        the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                        the coordinates changes)
        :im_width (int): The new image width
        :im_height (int): The new image height

    Returns:
    """

    right_padding = pad[0]
    bottom_padding = pad[1]

    if bottom_padding != 0:
        for aux in key_points:
            for point in aux:  # x 1 y 0
                y = point[0] * im_height
                point[0] = y / (im_height - pad[1])

    if right_padding != 0:
        for aux in key_points:
            for point in aux:
                x = point[1] * im_width
                point[1] = x / (im_width - pad[0])


def change_coordinates_aspect_ratio(aux_key_points_array, img_person, img_person_resized):
    """

    Args:
        :

    Returns:
        :
    """

    aux_key_points_array_ratio = []
    ratio_h, ratio_w = img_person.shape[0] / (img_person_resized.shape[1]), img_person.shape[1] / (img_person_resized.shape[2])  # shape 0 batch 1

    for elem in aux_key_points_array:
        aux = np.zeros(3)
        aux[0] = int((elem[0]) * ratio_h)
        aux[1] = int(elem[1] * ratio_h)
        aux[2] = int(elem[2])
        aux_key_points_array_ratio.append(aux)

    aux_key_points_array_ratio = np.array(aux_key_points_array_ratio, dtype=int)

    return aux_key_points_array_ratio


def parse_output_pose(heatmaps, offsets, threshold):
    """
    Parse the output pose (auxiliary function for tflite models)
    Args:
        :

    Returns:
        :
    """
    #
    # heatmaps: 9x9x17 probability of appearance of each keypoint in the particular part of the image (9,9) -> used to locate position of the joints
    # offsets: 9x9x34 used for calculation of the keypoint's position (first 17 x coords, the second 17 y coords)
    #
    joint_num = heatmaps.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(heatmaps.shape[-1]):
        joint_heatmap = heatmaps[..., i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)
        pose_kps[i, 0] = int(remap_pos[0] + offsets[max_val_pos[0], max_val_pos[1], i])
        pose_kps[i, 1] = int(remap_pos[1] + offsets[max_val_pos[0], max_val_pos[1], i + joint_num])
        max_prob = np.max(joint_heatmap)

        if max_prob > threshold:
            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                pose_kps[i, 2] = 1

    return pose_kps


def retrieve_xyz_from_detection(points_list, point_cloud_img):
    """
    Retrieve the xyz of the list of points passed as input (if we have the point cloud of the image)
    Args:
        :points_list (list): list of points for which we want to retrieve xyz information
        :point_cloud_img (numpy.ndarray): numpy array containing XYZRGBA information of the image

    Returns:
        :xyz (list): list of lists of 3D points with XYZ information (left camera origin (0,0,0))
    """

    xyz = [[point_cloud_img[:, :, 0][point[1], point[0]], point_cloud_img[:, :, 1][point[1], point[0]], point_cloud_img[:, :, 2][point[1], point[0]]]
           for point in points_list]
    return xyz


def retrieve_xyz_pose_points(point_cloud_image, key_points_score, key_points):
    """Retrieve the key points from the point cloud to get the XYZ position in the 3D space

    Args:
        :point_cloud_image (numpy.ndarray):
        :key_points_score (list):
        :key_points (list):

    Returns:
        :xyz_pose: a list of lists representing the XYZ 3D coordinates of each key point (j is the index number of the id pose)
    """
    xyz_pose = []

    for i in range(len(key_points_score)):
        xyz_pose_aux = []
        for j in range(len(key_points_score[i])):
            # if key_points_score[i][j] > threshold:# and j < 5:
            x, y = int(key_points[i][j][0] * point_cloud_image.shape[0]) - 1, int(key_points[i][j][1] * point_cloud_image.shape[1]) - 1
            xyz_pose_aux.append([point_cloud_image[x, y, 0], point_cloud_image[x, y, 1], point_cloud_image[x, y, 2], key_points_score[i][j]])

        xyz_pose.append(xyz_pose_aux)
    return xyz_pose


def compute_distance(points_list, min_distance=1.5):
    """
    Compute the distance between each point and find if there are points that are closer to each other that do not respect a certain distance
    expressed in meter.

    Args:
        :points_list (list): list of points expressed in xyz 3D coordinates (meters)
        :min_distance (float): minimum threshold for distances (if the l2 distance between two objects is lower than this value it is considered a violation)
            (default is 1.5)

    Returns:
        :distance_matrix: matrix containing the distances between each points (diagonal 0)
        :violate: set of points that violate the minimum distance threshold
        :couple_points: list of lists of couple points that violate the min_distance threshold (to keep track of each couple)
    """

    if points_list is None or len(points_list) == 1 or len(points_list) == 0:
        return None, None, None
    else:  # if there are more than two points
        violate = set()
        couple_points = []
        aux = np.array(points_list)
        distance_matrix = dist.cdist(aux, aux, 'euclidean')
        for i in range(0, distance_matrix.shape[0]):  # loop over the upper triangular of the distance matrix
            for j in range(i + 1, distance_matrix.shape[1]):
                if distance_matrix[i, j] < min_distance:
                    # print("Distance between {} and {} is {:.2f} meters".format(i, j, distance_matrix[i, j]))
                    violate.add(i)
                    violate.add(j)
                    couple_points.append((i, j))

        return distance_matrix, violate, couple_points


def initialize_video_recorder(output_path, output_depth_path, fps, shape):
    """Initialize OpenCV video recorders that will be used to write each image/frame to a single video

    Args:
        :output (str): The file location where the recorded video will be saved
        :output_depth (str): The file location where the recorded video with depth information will be saved
        :fps (int): The frame per seconds of the output videos
        :shape (tuple): The dimension of the output video (width, height)

    Returns:
        :writer (cv2.VideoWriter): The video writer used to save the video
        :writer_depth (cv2.VideoWriter): The video writer used to save the video with depth information
    """

    if not os.path.isdir(os.path.split(output_path)[0]):
        logger.error("Invalid path for the video writer; folder does not exist")
        exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, shape, True)
    writer_depth = None

    if output_depth_path:
        if not os.path.isdir(os.path.split(output_depth_path)[0]):
            logger.error("Invalid path for the depth video writer; folder does not exist")
            exit(1)
        writer_depth = cv2.VideoWriter(output_depth_path, fourcc, fps, shape, True)

    return writer, writer_depth


def delete_items_from_array_aux(arr, i):
    """
    Auxiliary function that delete the item at a certain index from a numpy array

    Args:
        :arr (numpy.ndarray): Array of array where each element correspond to the four coordinates of bounding box expressed in percentage
        :i (int): Index of the element to be deleted

    Returns:
        :arr_ret: the array without the element at index i
    """

    aux = arr.tolist()
    aux.pop(i)
    arr_ret = np.array(aux)
    return arr_ret


def fit_plane_least_square(xyz):
    # find a plane that best fit xyz points using least squares
    (rows, cols) = xyz.shape
    g = np.ones((rows, 3))
    g[:, 0] = xyz[:, 0]  # X
    g[:, 1] = xyz[:, 1]  # Y
    z = xyz[:, 2]
    (a, b, c), _, rank, s = np.linalg.lstsq(g, z, rcond=None)

    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)
    return d, normal, point


#
# def plot_plane(data, normal, d):
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#
#     # plot fitted plane
#     maxx = np.max(data[:, 0])
#     maxy = np.max(data[:, 1])
#     minx = np.min(data[:, 0])
#     miny = np.min(data[:, 1])
#
#     # compute needed points for plane plotting
#     xx, yy = np.meshgrid([minx - 10, maxx + 10], [miny - 10, maxy + 10])
#     z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
#
#     # plot plane
#     ax.plot_surface(xx, yy, z, alpha=0.2)
#
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()
#
#     return


def shape_to_np(shape, dtype="int"):
    """
    Function used for the dlib facial detector; it determine the facial landmarks for the face region, then convert the facial landmark
    (x, y)-coordinates to a NumPy array

    Args:
        :shape ():
        :dtype ():
            (Default is "int")

    Returns:
        :coordinates (list): list of x, y coordinates
    """
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coordinates


def rect_to_bb(rect):
    """
    Function used for the dlib facial detector; it converts dlib's rectangle to a tuple (x, y, w, h) where x and y represent xmin and ymin
    coordinates while w and h represent the width and the height

    Args:
        :rect (dlib.rectangle): dlib rectangle object that represents the region of the image where a face is detected

    Returns:
        :res (tuple): tuple that represents the region of the image where a face is detected in the form x, y, w, h
    """
    # take a bounding predicted by dlib and convert it to the format (x, y, w, h) as we would normally do with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    res = x, y, w, h
    return res


def enlarge_bb(y_min, x_min, y_max, x_max, im_width, im_height):
    """
    Enlarge the bounding box to include more background margin (used for face detection)

    Args:
        :y_min (int): the top y coordinate of the bounding box
        :x_min (int): the left x coordinate of the bounding box
        :y_max (int): the bottom y coordinate of the bounding box
        :x_max (int): the right x coordinate of the bounding box
        :im_width (int): The width of the image
        :im_height (int): The height of the image

    Returns:
        :y_min (int): the top y coordinate of the bounding box after enlarging
        :x_min (int): the left x coordinate of the bounding box after enlarging
        :y_max (int): the bottom y coordinate of the bounding box after enlarging
        :x_max (int): the right x coordinate of the bounding box after enlarging
    """

    y_min = int(max(0, y_min - abs(y_min - y_max) / 10))
    y_max = int(min(im_height, y_max + abs(y_min - y_max) / 10))
    x_min = int(max(0, x_min - abs(x_min - x_max) / 5))
    x_max = int(min(im_width, x_max + abs(x_min - x_max) / 4))  # 5
    x_max = int(min(x_max, im_width))
    return y_min, x_min, y_max, x_max


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]

    Args:
        :bb_test ():
        :bb_gt ():

    Returns:

    """
    # print(bb_test, bb_gt)
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (
            bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio

    Args:
        :bbox ():

    Returns:

    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h) if float(h) != 0 else w
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right

    Args:
        :x ():
        :score ():
            (Default is None)

    Returns:

    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers

    Args:
        :detections ():
        :trackers ():
        :iou_threshold ():
            (Default is 0.3)

    Returns:

    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    # print("IOU MATRIX: ", iou_matrix)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
        unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def find_face_from_key_points(key_points, bboxes, image, person=None, openpose=False, gazefollow=True):
    """

    Args:
        key_points:
        bboxes:
        image:
        person:
        openpose:
        gazefollow:

    Returns:

    """

    im_width, im_height = image.shape[1], image.shape[0]

    # key_points, bboxes = person.get_key_points()[-1], person.get_bboxes()[-1]
    # print("PERSON ID:", person.get_id())

    # 0 nose, 1/2 left/right eye, 3/4 left/right ear
    # 5/6	leftShoulder/rightShoulder
    # 7/8	leftElbow/rightElbow
    # 9/10	leftWrist/rightWrist
    # 11/12	leftHip/rightHip
    # 13/14	leftKnee/rightKnee
    # 15/16	leftAnkle/rightAnkle
    # print(key_points)

    face_points = key_points[:7]

    if openpose:
        face_points = []
        for point in key_points[:7]:
            # print(point[2], type(point[2]))
            if point[2] > 0.0:
                face_points.append(point)
    # print("face1", face_points)

    if len(face_points) == 0:
        return None, []

    # print("bboxe", bboxes, face_points)
    if not gazefollow:
        ct = compute_centroid(face_points)

        x_min, y_min = ct[0] - 10, ct[1] - 15
        x_max, y_max = ct[0] + 10, ct[1] + 10

        y_min_bbox = y_min

    elif gazefollow:
        # [l_shoulder, r_shoulder] = key_points[5:]
        # print(l_shoulder, r_shoulder)
        print("FACE", face_points)
        if len(face_points) == 1:
            return None, []

        x_min, y_min, _ = np.amin(face_points, axis=0)
        x_max, y_max, _ = np.amax(face_points, axis=0)

        # aux_diff =
        # print("X: ", aux_diff)
        # if aux_diff < 20:
        #     x_max += 20
        #     x_min -= 20

        aux_diff = y_max - y_min
        print("y: ", aux_diff)
        if aux_diff < 50:  # rapporto xmax -xmin o altro
            y_max += (x_max - x_min) / 1.4
            y_min -= (x_max - x_min) / 1.2
        # x_min -= 10
        # x_max += 10

        y_min_bbox = int(y_min)  # int(bboxes[1]) if bboxes is not None else y_min - (x_max-x_min)
        # if bboxes is None:
        #     y_max = y_max + (x_max-x_min)

    y_min, x_min, y_max, x_max = enlarge_bb(y_min_bbox, x_min, y_max, x_max, im_width, im_height)
    # print(y_min, x_min, y_max, x_max, y_max - y_min, x_max - x_min)
    # if -1 < y_max - y_min < 5 and -1 < x_max - x_min < 5:  # due punti uguali
    #     # print("AAAAA")
    #     return None, []

    face_image = image[y_min:y_max, x_min:x_max]

    if person is not None:
        # person.print_()
        person.update_faces(face_image)
        person.update_faces_coordinates([y_min, x_min, y_max, x_max])
        # person.update_faces_key_points(face_points)
        # person.print_()
        return None
    else:
        return face_image, [y_min, x_min, y_max, x_max]


def compute_interaction_cosine(head_position, target_position, gaze_direction):
    """
    Computes the interaction between two people using the angle of view.
    The interaction in measured as the cosine of the angle formed by the line from person A to B and the gaze direction of person A.

    Args:
        :head_position (list): list of pixel coordinates [x, y] that represents the position of the head of person A
        :target_position (list): list of pixel coordinates [x, y] that represents the position of head of person B
        :gaze_direction (list): list that represents the gaze direction of the head of person A in the form [gx, gy]

    Returns:
        :val (float): value that describe the quantity of interaction
    """

    if head_position == target_position:
        return 0  # or -1
    else:
        # direction from observer to target
        direction = np.arctan2((target_position[1] - head_position[1]), (target_position[0] - head_position[0]))
        direction_gaze = np.arctan2(gaze_direction[1], gaze_direction[0])
        difference = direction - direction_gaze

        # difference of the line joining observer -> target with the gazing direction,
        val = np.cos(difference)
        if val < 0:
            return 0
        else:
            return val


def compute_attention_from_vectors(list_objects):
    """

    Args:
        :list_objects ():

    Returns:

    """

    dict_person = dict()
    id_list = []
    for obj in list_objects:
        if len(obj.get_key_points()) > 0:
            # print("Object ID: ", obj.get_id(), "x: ", obj.get_poses_vector_norm()[-1][0], "y: ", obj.get_poses_vector_norm()[-1][1])
            id_list.append(obj.get_id())

            # print("kpts: ", obj.get_key_points()[-1])
            aux = [obj.get_key_points()[-1][j][:2] for j in [0, 2, 1, 4, 3]]
            dict_person[obj.get_id()] = [obj.get_poses_vector_norm()[-1], np.mean(aux, axis=0).tolist()]

    attention_matrix = np.zeros((len(dict_person), len(dict_person)), dtype=np.float32)

    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            if i == j:
                continue
            attention_matrix[i][j] = compute_interaction_cosine(dict_person[i][1], dict_person[j][1], dict_person[i][0])

    return attention_matrix.tolist(), id_list


def compute_attention_ypr(list_objects):
    """

    Args:
        :list_objects ():

    Returns:
        :
    """

    for obj in list_objects:
        if len(obj.get_key_points()) > 0:
            print("Object ID: ", obj.get_id(), "yaw: ", obj.get_poses_ypr()[-1][0], "pitch: ", obj.get_poses_ypr()[-1][1], "roll: ",
                  obj.get_poses_ypr()[-1][2])


def save_key_points_to_json(ids, kpts, path_json, openpose=False):
    """
    Save key points to .json format according to Openpose output format

    Args:
        :kpts ():
        :path_json ():

    Returns:
    """

    # print(path_json)
    dict_file = {"version": 1.3}
    list_dict_person = []
    for j in range(len(kpts)):
        dict_person = {"person_id": [int(ids[j])],
                       "face_keypoints_2d": [],
                       "hand_left_keypoints_2d": [],
                       "hand_right_keypoints_2d": [],
                       "pose_keypoints_3d": [],
                       "face_keypoints_3d": [],
                       "hand_left_keypoints_3d": [],
                       "hand_right_keypoints_3d": []}

        kpts_openpose = np.zeros((25, 3))
        for i, point in enumerate(kpts[j]):
            if openpose:
                idx_op = rev_pose_id_part_openpose[pose_id_part_openpose[i]]
            else:
                idx_op = rev_pose_id_part_openpose[pose_id_part[i]]
                # print(idx_op, point[1], point[0], point[2])
            kpts_openpose[idx_op] = [point[1], point[0], point[2]]  # x, y, conf

        list_kpts_openpose = list(np.concatenate(kpts_openpose).ravel())
        dict_person["pose_keypoints_2d"] = list_kpts_openpose
        # print(dict_person)
        list_dict_person.append(dict_person)

    dict_file["people"] = list_dict_person

    # Serializing json
    json_object = json.dumps(dict_file, indent=4)

    # Writing to sample.json
    with open(path_json, "w") as outfile:
        outfile.write(json_object)


def json_to_poses(json_data):
    """

    Args:
        :js_data ():

    Returns:
        :res ():
    """
    poses = []
    confidences = []
    ids = []

    for arr in json_data["people"]:
        ids.append(arr["person_id"])
        confidences.append(arr["pose_keypoints_2d"][2::3])
        aux = arr["pose_keypoints_2d"][2::3]
        arr = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
        # print("B", list(zip(arr[::2], arr[1::2])))
        poses.append(list(zip(arr[::2], arr[1::2], aux)))

    return poses, confidences, ids


def parse_json1(aux):
    # print(aux['people'])
    list_kpts = []
    id_list = []
    for person in aux['people']:
        # print(len(person['pose_keypoints_2d']))
        aux = person['pose_keypoints_2d']
        aux_kpts = [[aux[i+1], aux[i], aux[i+2]] for i in range(0, 75, 3)]
        # print(len(aux_kpts))
        list_kpts.append(aux_kpts)
        id_list.append(person['person_id'])

    # print(list_kpts)
    return list_kpts, id_list


def load_poses_from_json1(json_filename):
    """

    Args:
        :json_filename ():

    Returns:
        :poses, conf:
    """
    with open(json_filename) as data_file:
        loaded = json.load(data_file)
        zz = parse_json1(loaded)
        return zz


def load_poses_from_json(json_filename):
    """

    Args:
        :json_filename ():

    Returns:
        :poses, conf:
    """
    with open(json_filename) as data_file:
        loaded = json.load(data_file)
        poses, conf, ids = json_to_poses(loaded)

    if len(poses) < 1:  # != 1:
        return None, None, None
    else:
        return poses, conf, ids


def compute_head_features(img, pose, conf, open_pose=True):
    """

    Args:
        img:
        pose:
        conf:
        open_pose:

    Returns:

    """

    joints = [0, 15, 16, 17, 18] if open_pose else [0, 2, 1, 4, 3]

    n_joints_set = [pose[joint] for joint in joints if joint_set(pose[joint])]  # if open_pose else pose

    if len(n_joints_set) < 1:
        return None, None

    centroid = compute_centroid(n_joints_set)

    # for j in n_joints_set:
    #     print(j, centroid)
    max_dist = max([dist_2D([j[0], j[1]], centroid) for j in n_joints_set])

    new_repr = [(np.array([pose[joint][0], pose[joint][1]]) - np.array(centroid)) for joint in joints] if open_pose else [
        (np.array(pose[i]) - np.array(centroid)) for i in range(len(n_joints_set))]
    result = []

    for i in range(0, 5):

        if joint_set(pose[joints[i]]):
            if max_dist != 0.0:
                result.append([new_repr[i][0] / max_dist, new_repr[i][1] / max_dist])
            else:
                result.append([new_repr[i][0], new_repr[i][1]])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    conf_list = []

    for j in joints:
        conf_list.append(conf[j])

    return flat_list, conf_list, centroid


def compute_body_features(pose, conf):
    """

    Args:
        pose:
        conf:

    Returns:

    """
    joints = [0, 15, 16, 17, 18]
    alljoints = range(0, 25)

    n_joints_set = [pose[joint] for joint in joints if joint_set(pose[joint])]

    if len(n_joints_set) < 1:
        return None, None

    centroid = compute_centroid(n_joints_set)

    n_joints_set = [pose[joint] for joint in alljoints if joint_set(pose[joint])]

    max_dist = max([dist_2D(j, centroid) for j in n_joints_set])

    new_repr = [(np.array(pose[joint]) - np.array(centroid)) for joint in alljoints]

    result = []

    for i in range(0, 25):

        if joint_set(pose[i]):
            result.append([new_repr[i][0] / max_dist, new_repr[i][1] / max_dist])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    for j in alljoints:
        flat_list.append(conf[j])

    return flat_list, centroid


def compute_centroid(points):
    """

    Args:
        points:

    Returns:

    """
    x, y = [], []
    for point in points:
        if len(point) == 3:
            if point[2] > 0.0:
                x.append(point[0])
                y.append(point[1])
        else:
            x.append(point[0])
            y.append(point[1])

    # print(x, y)
    if x == [] or y == []:
        return [None, None]
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    return [mean_x, mean_y]


def joint_set(p):
    """

    Args:
        p:

    Returns:

    """
    return p[0] != 0.0 or p[1] != 0.0


def dist_2D(p1, p2):
    """

    Args:
        p1:
        p2:

    Returns:

    """
    # print(p1)
    # print(p2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def compute_head_centroid(pose):
    """

    Args:
        pose:

    Returns:

    """
    joints = [0, 15, 16, 17, 18]

    n_joints_set = [pose[joint] for joint in joints if joint_set(pose[joint])]

    # if len(n_joints_set) < 2:
    #     return None

    centroid = compute_centroid(n_joints_set)

    return centroid


def head_direction_to_json(path_json, norm_list, unc_list, ids_list, file_name):

    dict_file = {}
    list_dict_person = []
    for k, i in enumerate(norm_list):
        dict_person = {"id_person": [ids_list[k]],
                       "norm_xy": [i[0][0].item(), i[0][1].item()],  # from numpy to native python type for json serilization
                       "center_xy": [int(i[1][0]), int(i[1][1])],
                       "uncertainty": [unc_list[k].item()]}

        list_dict_person.append(dict_person)
    dict_file["people"] = list_dict_person

    json_object = json.dumps(dict_file, indent=4)

    with open(path_json, "w") as outfile:
        outfile.write(json_object)


def ypr_to_json(path_json, yaw_list, pitch_list, roll_list, yaw_u_list, pitch_u_list, roll_u_list, ids_list, center_xy):

    dict_file = {}
    list_dict_person = []
    for k in range(len(yaw_list)):
        dict_person = {"id_person": [ids_list[k]],
                       "yaw": [yaw_list[k].item()],
                       "yaw_u": [yaw_u_list[k].item()],
                       "pitch": [pitch_list[k].item()],
                       "pitch_u": [pitch_u_list[k].item()],
                       "roll": [roll_list[k].item()],
                       "roll_u": [roll_u_list[k].item()],
                       "center_xy": [int(center_xy[k][0]), int(center_xy[k][1])]}

        list_dict_person.append(dict_person)
    dict_file["people"] = list_dict_person

    json_object = json.dumps(dict_file, indent=4)

    with open(path_json, "w") as outfile:
        outfile.write(json_object)
    # exit()


def save_keypoints_image(img, poses, suffix_, path_save=''):
    """
    Save the image with the key points drawn on it
    Args:
        img:
        poses:
        suffix_:

    Returns:

    """
    aux = img.copy()
    for point in poses:
        for i, p in enumerate(point):
            if i in [0, 15, 16, 17, 18]:
                cv2.circle(aux, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(path_save, suffix_ + '.jpg'), aux)


def unit_vector(vector):
    """
    Returns the unit vector of the vector.

    Args:
        vector:

    Returns:

    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    # if not unit vector
    v1_u = unit_vector(tuple(v1))
    v2_u = unit_vector(tuple(v2))
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle if angle < 1.80 else angle - 1.80


def centroid_constraint(centroid, centroid_det, gazefollow=False):  # x y
    """

    Args:
        centroid:
        centroid_det:

    Returns:

    """
    if centroid_det == [None, None]:
        return False

    if gazefollow == False:
        if 0 < centroid_det[0] < 143 and 0 < centroid_det[1] < 24:  # centroid in the overprinted text of hour in the video
            return False
        if 0 < centroid_det[1] < 4:
            return False
        if centroid[0] - 3 < centroid_det[0] < centroid[0] + 3 and centroid[1] - 3 < centroid_det[1] < centroid[
            1] + 3:  # detected centroid near the gt centroid
            return True
        else:
            return False
    else:
        if int(centroid[0] - 30) < int(centroid_det[0]) < int(centroid[0] + 30) and int(centroid[1] - 30) < int(centroid_det[1]) < int(
                centroid[1] + 30):  # detected centroid near the gt centroid
            return True
        else:
            return False


def initialize_video_reader(path_video):
    """

    Args:
        path_video:

    Returns:

    """
    cap = cv2.VideoCapture(path_video)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', path_video)
        exit(-1)
    return cap


def distance_skeletons(kpts1, kpts2, dst_type):
    """
    Function to compute the distance between skeletons
    #TO DO
    Args:
        kpts1:
        kpts2:
        dts_type:

    Returns:

    """
    if len(kpts1) != len(kpts2):
        print("Error: Different notation used for keypoints")
        exit(-1)

    print(len(kpts1), len(kpts2))
    # to openpose notations
    if len(kpts1) == len(kpts2) == 17:
        kpts1, kpts2 = kpt_centernet_to_openpose(kpts1), kpt_centernet_to_openpose(kpts2)
    print(len(kpts1), len(kpts2))

    if len(kpts1) != 25 or len(kpts2) != 25:
        print("Error")
        exit(-1)

    res_dist = 0

    if dst_type == 'all_points':
        for i, _ in enumerate(kpts1):
            res_dist += dist_2D(kpts1[i][:2], kpts2[i][:2])
        res_dist /= 25
        return res_dist

    elif dst_type == 'head_centroid':
        top1_c, top2_c = compute_head_centroid(kpts1), compute_head_centroid(kpts2)
        if top1_c == [None, None] or top2_c == [None, None]:
            res_dist = 900
        else:
            res_dist = dist_2D(top1_c[:2], top2_c[:2])
        return res_dist

    elif dst_type == 'three_centroids':
        #TO DO
        # top1_c, top2_c = compute_centroid(kpts1[0, 15, 16, 17, 18]), compute_centroid(kpts2[0, 15, 16, 17, 18])
        # mid1_c, mid2_c = compute_centroid(kpts1[2, 5, 9, 12]), compute_centroid(kpts2[2, 5, 9, 12])
        # btm1_c, btm2_c = compute_centroid(kpts1[9, 12, 10, 13]), compute_centroid(kpts2[9, 12, 10, 13])
        # res_dist = dist_2D(top1_c[:2], top2_c[:2]) + dist_2D(mid1_c[:2], mid2_c[:2]) + dist_2D(btm1_c[:2], btm2_c[:2])
        # res_dist /= 3
        # return res_dist
        return None

    elif dst_type == '':
        print("dst_typ not valid")
        exit(-1)


def kpt_openpose_to_centernet(kpts):
    """

    Args:
        kpts:

    Returns:

    """
    #TO TEST
    kpts_openpose = np.zeros((16, 3))
    for i, point in enumerate(kpts):
        idx_op = rev_pose_id_part[pose_id_part_openpose[i]]
        kpts_openpose[idx_op] = [point[0], point[1], point[2]]

    return kpts_openpose


def kpt_centernet_to_openpose(kpts):
    """

    Args:
        kpts:

    Returns:

    """
    #TO TEST
    kpts_openpose = np.zeros((25, 3))
    for i, point in enumerate(kpts):
        idx_op = rev_pose_id_part_openpose[pose_id_part[i]]
        kpts_openpose[idx_op] = [point[1], point[0], point[2]]

    return kpts_openpose


def non_maxima_aux(det, kpt, threshold=15):  # threshold in pxels
    # print("A", kpt, "\n", len(kpt))

    indexes_to_delete = []

    if len(kpt) == 0 or len(det) == 0:
        return [], []

    if len(kpt) == 1 or len(det) == 1:
        return det, kpt

    kpt_res = kpt.copy()
    det_res_aux = det.copy()

    for i in range(0, len(kpt)):
        for j in range(i, len(kpt)):
            if i == j:
                continue
            dist = distance_skeletons(kpt[i], kpt[j], 'head_centroid')
            # print("DIST", i, j, dist)
            if dist < threshold:
                if j not in indexes_to_delete:
                    indexes_to_delete.append(j)
                # kpt_res.pop(j)
    det_res = []

    # print(indexes_to_delete)
    indexes_to_delete = sorted(indexes_to_delete, reverse=True)
    # print(len(kpt_res))
    for index in indexes_to_delete:
        kpt_res.pop(index)

    det_res_aux = list(np.delete(det_res_aux, indexes_to_delete, axis=0))
    det_res = np.array(det_res_aux)

    return det_res, kpt_res


def compute_centroid_list(points):
    """

    Args:
        points:

    Returns:

    """
    x, y = [], []
    for i in range(0, len(points), 3):
        if points[i + 2] > 0.0:  # confidence openpose
            x.append(points[i])
            y.append(points[i + 1])

    if x == [] or y == []:
        return [None, None]
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    return [mean_x, mean_y]


def normalize_wrt_maximum_distance_point(points, file_name=''):
    centroid = compute_centroid_list(points)
    # centroid = [points[0], points[1]]
    # print(centroid)
    # exit()

    max_dist_x, max_dist_y = 0, 0
    for i in range(0, len(points), 3):
        if points[i + 2] > 0.0:  # confidence openpose take only valid keypoints (if not detected (0, 0, 0)
            distance_x = abs(points[i] - centroid[0])
            distance_y = abs(points[i+1] - centroid[1])
            # dist_aux.append(distance)
            if distance_x > max_dist_x:
                max_dist_x = distance_x
            if distance_y > max_dist_y:
                max_dist_y = distance_y
        elif points[i + 2] == 0.0: # check for centernet people on borders with confidence 0
            points[i] = 0
            points[i+1] = 0

    for i in range(0, len(points), 3):
        if points[i + 2] > 0.0:
            if max_dist_x != 0.0:
                points[i] = (points[i] - centroid[0]) / max_dist_x
            if max_dist_y != 0.0:
                points[i + 1] = (points[i + 1] - centroid[1]) / max_dist_y
            if max_dist_x == 0.0:  # only one point valid with some confidence value so it become (0,0, confidence)
                points[i] = 0.0
            if max_dist_y == 0.0:
                points[i + 1] = 0.0

    return points


def retrieve_interest_points(kpts, detector):
    """

    :param kpts:
    :return:
    """
    res_kpts = []

    if detector == 'centernet':
        face_points = [0, 1, 2, 3, 4]
        for index in face_points:
            res_kpts.append(kpts[index][1])
            res_kpts.append(kpts[index][0])
            res_kpts.append(kpts[index][2])
    elif detector== 'zedcam':
        face_points = [0, 14, 15, 16, 17]
        for index in face_points:
            res_kpts.append(kpts[index][0])
            res_kpts.append(kpts[index][1])
            res_kpts.append(kpts[index][2])
    else:
        # take only interest points (5 points of face)
        face_points = [0, 16, 15, 18, 17]
        for index in face_points:
            res_kpts.append(kpts[index][0])
            res_kpts.append(kpts[index][1])
            res_kpts.append(kpts[index][2])



    return res_kpts

def create_bbox_from_openpose_keypoints(data):
    # from labels import pose_id_part_openpose
    bbox = list()
    ids = list()
    kpt = list()
    kpt_scores = list()
    for person in data['people']:
        ids.append(person['person_id'][0])
        kpt_temp = list()
        kpt_score_temp = list()
        # create bbox with min max each dimension
        x, y = [], []
        for i in pose_id_part_openpose:
            if i < 25:
                # kpt and kpts scores
                kpt_temp.append([int(person['pose_keypoints_2d'][i * 3]), int(person['pose_keypoints_2d'][(i * 3) + 1]),
                                 person['pose_keypoints_2d'][(i * 3) + 2]])
                kpt_score_temp.append(person['pose_keypoints_2d'][(i * 3) + 2])
                # check confidence != 0
                if person['pose_keypoints_2d'][(3 * i) + 2]!=0:
                    x.append(int(person['pose_keypoints_2d'][3 * i]))
                    y.append(int(person['pose_keypoints_2d'][(3 * i) + 1]))
        kpt_scores.append(kpt_score_temp)
        kpt.append(kpt_temp)
        xmax = max(x)
        xmin = min(x)
        ymax = max(y)
        ymin = min(y)
        bbox.append([xmin, ymin, xmax, ymax, 1])  # last value is for compatibility of centernet

    return bbox, kpt, kpt_scores  # not to use scores

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
           alist.sort(key=natural_keys) sorts in human order
           http://nedbatchelder.com/blog/200712/human_sorting.html
           (See Toothy's implementation in the comments)
           """
    import re
    return [atoi(c) for c in re.split(r'(\d+)', text)]