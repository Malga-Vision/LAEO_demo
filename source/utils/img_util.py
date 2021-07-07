import cv2
import os
import json
import numpy as np
from math import cos, sin, pi
from utils.labels import coco_category_index, rgb_colors, color_pose, color_pose_normalized, pose_id_part, face_category_index, body_parts_openpose, body_parts, face_points, face_points_openpose, pose_id_part_zedcam, face_points_zedcam, body_parts_zedcam
# from src.utils.my_utils import fit_plane_least_square  # , retrieve_line_from_two_points


def percentage_to_pixel(shape, bb_boxes, bb_boxes_scores, key_points=None, key_points_score=None):
    """
    Convert the detections from percentage to pixels coordinates; it works both for the bounding boxes and for the key points if passed

    Args:
        :img_shape (tuple): the shape of the image
        :bb_boxes (numpy.ndarray): list of list each one representing the bounding box coordinates expressed in percentage [y_min_perc, x_min_perc, y_max_perc, x_max_perc]
        :bb_boxes_scores (numpy.ndarray): list of score for each bounding box in range [0, 1]
        :key_points (numpy.ndarray): list of list of list each one representing the key points coordinates expressed in percentage [y_perc, x_perc]
        :key_points_score (numpy.ndarray): list of list each one representing the score associated to each key point in range [0, 1]

    Returns:
        :det (numpy.ndarray): list of lists each one representing the bounding box coordinates in pixels and the score associated to each bounding box [x_min, y_min, x_max, y_max, score]
        :kpt (list): list of lists each one representing the key points detected in pixels and the score associated to each point [x, y, score]
    """

    im_width, im_height = shape[1], shape[0]
    det, kpt = [], []

    if key_points is not None:
        key_points = key_points
        key_points_score = key_points_score

    for i, _ in enumerate(bb_boxes):
        y_min, x_min, y_max, x_max = bb_boxes[i]
        x_min_rescaled, x_max_rescaled, y_min_rescaled, y_max_rescaled = x_min * im_width, x_max * im_width, y_min * im_height, y_max * im_height
        det.append([int(x_min_rescaled), int(y_min_rescaled), int(x_max_rescaled), int(y_max_rescaled), bb_boxes_scores[i]])

        if key_points is not None:
            aux_list = []
            for n, key_point in enumerate(key_points[i]):  # y x
                aux = [int(key_point[0] * im_height), int(key_point[1] * im_width), key_points_score[i][n]]
                aux_list.append(aux)
            kpt.append(aux_list)

    det = np.array(det)

    return det, kpt


def draw_detections(image, detections, max_boxes_to_draw, violate=None, couple_points=None, draw_class_score=False):
    """
    Given an image and a dictionary of detections this function return the image with the drawings of the bounding boxes (with violations information if specified)

    Args:
        :img (numpy.ndarray): The image that is given as input to the object detection model
        :detections (dict): The dictionary with the detections information (detection_classes, detection_boxes, detection_scores,
            detection_keypoint_scores, detection_keypoints, detection_boxes_centroid)
        :max_boxes_to_draw (int): The maximum number of bounding boxes to draw
        :violate (set): The indexes of detections (sorted) that violate the minimum distance computed by my_utils.compute_distance function
            (default is None)
        :couple_points (list): A list of tuples each one containing the couple of indexes that violate the minimum distance (used to draw lines in
            between to bounding boxes)
            (default is None)
        :draw_class_score (bool): If this value is set to True, in the returned image will be drawn the category and the score over each bounding box
            (default is False)

    Returns:
        :img_with_drawings (numpy.ndarray): The image with the bounding boxes of each detected objects and optionally with the situations of violation
    """

    im_width, im_height = image.shape[1], image.shape[0]
    img_with_drawings = image.copy()
    classes = detections['detection_classes']
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    centroids = detections['detection_boxes_centroid']
    red = (0, 0, 255)

    i = 0
    while i < max_boxes_to_draw and i < len(classes):
        [y_min, x_min, y_max, x_max] = boxes[i]
        (x_min_rescaled, x_max_rescaled, y_min_rescaled, y_max_rescaled) = (x_min * im_width, x_max * im_width, y_min * im_height, y_max * im_height)
        start_point, end_point = (int(x_max_rescaled), int(y_max_rescaled)), (int(x_min_rescaled), int(y_min_rescaled))

        # [cx, cy] = centroids[i]
        # (cx_rescaled, cy_rescaled) = (int(cx * im_width), int(cy * im_height))

        color = rgb_colors[classes[i]]
        if violate:
            if i in violate:
                color = red

        cv2.rectangle(img_with_drawings, start_point, end_point, color, 2)
        # cv2.circle(img_with_drawings, (cx_rescaled, cy_rescaled), 2, color, 2)

        if draw_class_score:
            cv2.rectangle(img_with_drawings, end_point, (start_point[0], end_point[1] - 25), rgb_colors[classes[i]], -1)
            text = face_category_index[classes[i]]['name'] + " {:.2f}".format(scores[i])
            cv2.putText(img_with_drawings, text, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        i += 1

    if couple_points and len(centroids) > 1:
        for j in range(len(couple_points)):
            pt1 = centroids[couple_points[j][0]][0], centroids[couple_points[j][0]][1]
            pt2 = centroids[couple_points[j][1]][0], centroids[couple_points[j][1]][1]
            cv2.line(img_with_drawings, pt1, pt2, red, 2)

    text_location = (int(image.shape[1]-image.shape[1]/4), int(image.shape[0]/17))
    font_scale = 0.8 * 1 / (640/image.shape[0])
    thickness = int(2 * (image.shape[0]/640))
    cv2.putText(img_with_drawings, "# of people : "+str(i), text_location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, red, thickness, cv2.LINE_AA)

    return img_with_drawings


def resize_preserving_ar(image, new_shape):
    """
    Resize and pad the input image in order to make it usable by an object detection model (e.g. mobilenet 640x640)

    Args:
        :image (numpy.ndarray): The image that will be resized and padded
        :new_shape (tuple): The shape of the image output (height, width)

    Returns:
        :res_image (numpy.ndarray): The image modified to have the new shape
    """
    (old_height, old_width, _) = image.shape
    (new_height, new_width) = new_shape

    if old_height != old_width:  # rectangle
        ratio_h, ratio_w = new_height / old_height, new_width / old_width

        if ratio_h > ratio_w:
            dim = (new_width, int(old_height * ratio_w))
            img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            bottom_padding = int(new_height - int(old_height * ratio_w)) if int(new_height - int(old_height * ratio_w)) >= 0 else 0
            img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT)
            pad = (0, bottom_padding, dim)

        else:
            dim = (int(old_width * ratio_h), new_height)
            img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            right_padding = int(new_width - int(old_width * ratio_h)) if int(new_width - int(old_width * ratio_h)) >= 0 else 0
            img = cv2.copyMakeBorder(img, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT)
            pad = (right_padding, 0, dim)

    else:  # square
        img = cv2.resize(image, new_shape, new_height, new_width)
        pad = (0, 0, (new_height, new_width))

    return img, pad


def resize_and_padding_preserving_ar(image, new_shape):
    """ Resize and pad the input image in order to make it usable by a pose model (e.g. mobilenet-posenet takes as input 257x257 images)

    Args:
        :image (numpy.ndarray): The image that will be resized and padded
        :new_shape (tuple): The shape of the image output

    Returns:
        :res_image (numpy.ndarray): The image modified to have the new shape
    """

    (old_height, old_width, _) = image.shape
    (new_height, new_width) = new_shape

    if old_height != old_width:  # rectangle
        ratio_h, ratio_w = new_height / old_height, new_width / old_width

        # print(img.shape, "\nRATIO: ", ratio_h, ratio_w)
        if ratio_h < ratio_w:
            ratio = new_shape[0] / old_height
            dim = (int(old_width * ratio), new_width)
            img = cv2.resize(image, dim)
            right_padding = int(new_width - img.shape[1]) if int(new_width - img.shape[1]) >= 0 else 0
            img = cv2.copyMakeBorder(img, 0, 0, 0, right_padding, cv2.BORDER_CONSTANT)
        else:
            ratio = new_shape[1] / old_width
            dim = (new_height, int(old_height * ratio))
            img = cv2.resize(image, dim)
            bottom_padding = int(new_height - img.shape[0]) if int(new_width - img.shape[0]) >= 0 else 0
            img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT)

    else:  # square
        img = cv2.resize(image, new_shape)

    img = img.astype(np.float32) / 255.
    res_image = np.expand_dims(img, 0)

    return res_image


def draw_axis(yaw, pitch, roll, image=None, tdx=None, tdy=None, size=50):
    """
    Draw yaw pitch and roll axis on the image if passed as input and returns the vector containing the projection of the vector on the image plane

    Args:
        :yaw (float): value that represents the yaw rotation of the face
        :pitch (float): value that represents the pitch rotation of the face
        :roll (float): value that represents the roll rotation of the face
        :image (numpy.ndarray): The image where the three vector will be printed
            (default is None)
        :tdx (float64): x coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :tdy (float64): y coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :size (int): value that will be multiplied to each x, y and z value that enlarge the "vector drawing"
            (default is 50)

    Returns:
        :list_projection_xy (list): list containing the unit vector [x, y, z]
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy

    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # PROJECT 3D TO 2D XY plane (Z = 0)

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    z3 = size * (cos(pitch) * cos(yaw)) + tdy

    if image is not None:
        cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    list_projection_xy = [sin(yaw), -cos(yaw) * sin(pitch)]
    return list_projection_xy


def visualize_vector(image, center, unit_vector, title="", color=(0, 0, 255)):
    """
    Draw the projected vector on the image plane and return the image

    Args:
        :image (numpy.ndarray): The image where the vector will be printed
        :center (list): x, y coordinates in pixels of the starting point from where the vector is drawn
        :unit_vector (list): vector of the gaze in the form [gx, gy]
        :title (string): title displayed in the imshow function
            (default is "")
        :color (tuple): color value of the vector drawn on the image
            (default is (0, 0, 255))

    Returns:
        :result (numpy.ndarray): The image with the vectors drawn
    """
    unit_vector_draw = [unit_vector[0] * image.shape[0]*0.15, unit_vector[1] * image.shape[0]*0.15]
    point = [center[0] + unit_vector_draw[0], center[1] + unit_vector_draw[1]]

    result = cv2.arrowedLine(image, (int(center[0]), int(center[1])), (int(point[0]), int(point[1])), color, thickness=2, tipLength=0.2)

    return result


def draw_key_points_pose(image, kpt, openpose=False):
    """
    Draw the key points and the lines connecting them; it expects the output of CenterNet (not OpenPose format)

    Args:
        :image (numpy.ndarray): The image where the lines connecting the key points will be printed
        :kpt (list): list of lists of points detected for each person [[x1, y1, c1], [x2, y2, c2],...] where x and y represent the coordinates of each
            point while c represents the confidence

    Returns:
        :img (numpy.ndarray): The image with the drawings of lines and key points
    """

    parts = body_parts_openpose if openpose else body_parts
    kpt_score = None
    threshold = 0.4

    overlay = image.copy()

    face_pts = face_points_openpose if openpose else face_points

    for j in range(len(kpt)):
        # 0 nose, 1/2 left/right eye, 3/4 left/right ear
        color = color_pose["blue"]
        if j == face_pts[0]:  # naso
            color = color_pose["purple"]
        if j == face_pts[1]:
            color = color_pose["light_pink"]
        if j == face_pts[2]:
            color = color_pose["dark_pink"]
        if j == face_pts[3]:
            color = color_pose["light_orange"]
        if j == face_pts[4]:
            color = color_pose["dark_orange"]
        if openpose:
            cv2.circle(image, (int(kpt[j][0]), int(kpt[j][1])), 1, color, 2)
        else:
            cv2.circle(image, (int(kpt[j][1]), int(kpt[j][0])), 1, color, 2)
        # cv2.putText(img, pose_id_part[i], (int(kpts[j][i, 1] * img.shape[1]), int(kpts[j][i, 0] * img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    for part in parts:
        if int(kpt[part[0]][1]) != 0 and int(kpt[part[0]][0]) != 0 and int(kpt[part[1]][1]) != 0 and int(
                kpt[part[1]][0]) != 0:

            if openpose:
                cv2.line(overlay, (int(kpt[part[0]][0]), int(kpt[part[0]][1])), (int(kpt[part[1]][0]), int(kpt[part[1]][1])), (255, 255, 255), 2)
            else:
                cv2.line(overlay, (int(kpt[part[0]][1]), int(kpt[part[0]][0])),
                         (int(kpt[part[1]][1]), int(kpt[part[1]][0])), (255, 255, 255), 2)

    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image

def draw_key_points_pose_zedcam(image, kpt, openpose=True):
    """
    Draw the key points and the lines connecting them; it expects the output of CenterNet (not OpenPose format)

    Args:
        :image (numpy.ndarray): The image where the lines connecting the key points will be printed
        :kpt (list): list of lists of points detected for each person [[x1, y1, c1], [x2, y2, c2],...] where x and y represent the coordinates of each
            point while c represents the confidence

    Returns:
        :img (numpy.ndarray): The image with the drawings of lines and key points
    """

    parts = body_parts_zedcam
    kpt_score = None
    threshold = 0.4

    overlay = image.copy()

    face_pts = face_points_zedcam

    for j in range(len(kpt)):
        # 0 nose, 1/2 left/right eye, 3/4 left/right ear
        color = color_pose["blue"]
        if j == face_pts[0]:  # naso
            color = color_pose["purple"]
        if j == face_pts[1]:
            color = color_pose["light_pink"]
        if j == face_pts[2]:
            color = color_pose["dark_pink"]
        if j == face_pts[3]:
            color = color_pose["light_orange"]
        if j == face_pts[4]:
            color = color_pose["dark_orange"]
        if openpose:
            cv2.circle(image, (int(kpt[j][0]), int(kpt[j][1])), 1, color, 2)
        else:
            cv2.circle(image, (int(kpt[j][1]), int(kpt[j][0])), 1, color, 2)
        # cv2.putText(img, pose_id_part[i], (int(kpts[j][i, 1] * img.shape[1]), int(kpts[j][i, 0] * img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    for part in parts:
        if int(kpt[part[0]][1]) != 0 and int(kpt[part[0]][0]) != 0 and int(kpt[part[1]][1]) != 0 and int(
                kpt[part[1]][0]) != 0:

            if openpose:
                cv2.line(overlay, (int(kpt[part[0]][0]), int(kpt[part[0]][1])), (int(kpt[part[1]][0]), int(kpt[part[1]][1])), (255, 255, 255), 2)
            else:
                cv2.line(overlay, (int(kpt[part[0]][1]), int(kpt[part[0]][0])),
                         (int(kpt[part[1]][1]), int(kpt[part[1]][0])), (255, 255, 255), 2)

    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image

def plot_3d_points(list_points):
    """
    Plot points in 3D

    Args:
        :list_points: A list of lists representing the points; each point has (x, y, z) coordinates represented by the first, second and third element of each list

    Returns:
    """
    if list_points == []:
        return

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point in list_points:
        ax.scatter(point[0], point[1], point[2], c=np.array(0), marker='o')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    return


def draw_on_img(image, center, id_, res):
    """
    Draw arrow illustrating gaze direction on the image

    Args:
        :image (numpy.ndarray): The image where the vector will be printed
        :center (list): x, y coordinates in pixels of the starting point from where the vector is drawn
        :id_ (string): title displayed in the imshow function
            (default is "")
        :res (list): vector of the gaze in the form [gx, gy]

    Returns:
        :img_arrow (numpy.ndarray): The image with the vector drawn
    """

    res[0] *= image.shape[0]
    res[1] *= image.shape[1]

    norm1 = res / np.linalg.norm(res)
    norm_aux = [norm1[0], norm1[1]]  # normalized vectors

    norm1[0] *= image.shape[0]*0.15
    norm1[1] *= image.shape[0]*0.15

    point = center + norm1


    img_arrow = cv2.arrowedLine(image.copy(), (int(center[1]), int(center[0])), (int(point[1]), int(point[0])), (0, 0, 255), thickness=2, tipLength=0.2)

    return img_arrow, [norm_aux, center]


def confusion_matrix(conf_matrix, target_names=None, title="", cmap=None):
    """
    Create the image of the confusion matrix given a matrix as input

    Args:
        :conf_matrix (list): list of lists that represent an MxM matrix e.g. [[v11, v12, v13], [v21, v22, v23], [v31, v32, v33]]
        :target_names (list): list of target name of dimension M e.g. [[label1, label2, label3]]
            (default is None)
        :title (string): title string to be printed in the confusion matrix
            (default is "")
        :cmap (string): colormap that will be used by the confusion matrix
            (default is None)

    Returns:
        :gbr (numpy.ndarray): The image where the lines connecting the key points will be printed
    """
    from laeo_per_frame.interaction_per_frame_uncertainty import LAEO_computation
    import matplotlib.pyplot as plt

    if not conf_matrix:
        return []

    # if cmap is None:
    #     cmap = plt.get_cmap('Blues')

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    fig, ax = plt.subplots(figsize=(6, 4))  # 2, 2, figsize=(6, 4))
    cax = ax.imshow(conf_matrix)

    for i in range(len(conf_matrix[0])):
        for j in range(len(conf_matrix[1])):
            ax.text(j, i, str(np.around(conf_matrix[i][j], 3)), va='center', ha='center', color="black")

    if target_names is not None:
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.colorbar(cax)
    # plt.show()

    fig.canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    aux_img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    gbr = aux_img[..., [2, 0, 1]].copy()

    # cv2.imshow("1312", gbr)
    # cv2.waitKey(0)

    return gbr


def join_images(image1, image2):
    """
    Join two images vertically into a new image with the height that is the maximum height of the two images passed as input and the width that is
    the sum of the widths of the two images passed as input

    Args:
        :image1 (numpy.ndarray): The image that will be in the left part of the joined images
        :image2 (numpy.ndarray): The image that will be in the right part of the joined images

    Returns:
        :joined_image (numpy.ndarray): The image that is the results of the merge of the two images passed as input
    """

    if type(image1) == list or type(image2) == list:
        return None

    image1_width, image1_height, image2_width, image2_height = image1.shape[1], image1.shape[0], image2.shape[1], image2.shape[0]

    new_shape_height = max(image1_height, image2_height)
    new_shape = (new_shape_height, image1_width + image2_width, 3)

    joined_image = np.zeros(new_shape, dtype=np.uint8)
    joined_image[:image1_height, :image1_width, :] = image1
    joined_image[:image2_height, image1_width:, :] = image2

    cv2.imshow("", cv2.resize(joined_image, (1200, 500)))
    cv2.waitKey(0)
    return joined_image


def draw_axis_from_json(img, json_file):
    if os.path.isfile(json_file):
        cv2.imshow("", img)
        cv2.waitKey(0)

        with open(json_file) as f:
            data = json.load(f)
            print(data)
            aux = data['people']
            for elem in aux:
                draw_axis(elem['yaw'][0], elem['pitch'][0], elem['roll'][0], img, elem['center_xy'][0], elem['center_xy'][1])
        cv2.imshow("", img)
        cv2.waitKey(0)

    return


def points_on_circumference(center=(0, 0), r=50, n=100):
    return [(center[0] + (cos(2 * pi / n * x) * r), center[1] + (sin(2 * pi / n * x) * r)) for x in range(0, n + 1)]


def draw_cones(yaw, pitch, roll, unc_yaw, unc_pitch, unc_roll, image=None, tdx=None, tdy=None, size=300):
    """
    Draw yaw pitch and roll axis on the image if passed as input and returns the vector containing the projection of the vector on the image plane

    Args:
        :yaw (float): value that represents the yaw rotation of the face
        :pitch (float): value that represents the pitch rotation of the face
        :roll (float): value that represents the roll rotation of the face
        :image (numpy.ndarray): The image where the three vector will be printed
            (default is None)
        :tdx (float64): x coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :tdy (float64): y coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :size (int): value that will be multiplied to each x, y and z value that enlarge the "vector drawing"
            (default is 50)

    Returns:
        :list_projection_xy (list): list containing the unit vector [x, y, z]
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy

    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # PROJECT 3D TO 2D XY plane (Z = 0)

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    z3 = size * (cos(pitch) * cos(yaw)) + tdy

    unc_mean = (unc_yaw + unc_pitch + unc_roll) / 3

    radius = 12 * unc_mean

    overlay = image.copy()
    if image is not None:
        # cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        # cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        points = points_on_circumference((int(x3), int(y3)), radius, 400)

        for point in points:
            cv2.line(image, (int(tdx), int(tdy)), (int(point[0]), int(point[1])), (255, 0, 0), 2)

        # cv2.circle(image, (int(x3), int(y3)), int(radius), (255, 0, 0), 2)

    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # cv2.imshow("cc", image)
    # cv2.waitKey(0)
    # exit()

    list_projection_xy = [sin(yaw), -cos(yaw) * sin(pitch)]
    return list_projection_xy, image

def draw_axis_3d(yaw, pitch, roll, image=None, tdx=None, tdy=None, size=50, yaw_uncertainty=-1, pitch_uncertainty=-1, roll_uncertainty=-1):
    """
    Draw yaw pitch and roll axis on the image if passed as input and returns the vector containing the projection of the vector on the image plane
    Args:
        :yaw (float): value that represents the yaw rotation of the face
        :pitch (float): value that represents the pitch rotation of the face
        :roll (float): value that represents the roll rotation of the face
        :image (numpy.ndarray): The image where the three vector will be printed
            (default is None)
        :tdx (float64): x coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :tdy (float64): y coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :size (int): value that will be multiplied to each x, y and z value that enlarge the "vector drawing"
            (default is 50)
    Returns:
        :list_projection_xy (list): list containing the unit vector [x, y, z]
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    # print(yaw, pitch, roll)
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # PROJECT 3D TO 2D XY plane (Z = 0)
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    z3 = size * (cos(pitch) * cos(yaw)) + tdy
    if image is not None:
        cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
    return image