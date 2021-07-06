from utils.my_utils import rescale_bb, rescale_key_points, delete_items_from_array_aux, enlarge_bb
from utils.labels import coco_category_index, face_category_index
import time
import numpy as np


def detect(model, image, min_score_thresh, new_old_shape):
    """
    Detect objects in the image running the model

    Args:
        :model (tensorflow.python.saved_model): The Tensorflow object detection model
        :image (numpy.ndarray): The image that is given as input to the object detection model
        :min_score_threshold (float): The minimum score for the detections (detections with a score lower than this value will be discarded)
        :new_old_shape (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                                the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                                the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                                the coordinates changes that we have to do)

    Returns:
        :detections (dict): dictionary with detection scores, classes, centroids and bounding box coordinates ordered by score in descending order
        :inference_time (float): inference time for one image expressed in seconds
    """
    image = np.array(image).astype(np.uint8)
    input_tensor = np.expand_dims(image, axis=0)

    start_time = time.time()
    det = model(input_tensor)
    end_time = time.time()

    detections = filter_detections(det, min_score_thresh, image.shape, new_old_shape)
    inference_time = end_time - start_time
    return detections, inference_time


def filter_detections(detections, min_score_thresh, shape, new_old_shape=None):
    """
    Filter the detections based on a minimum threshold value and modify the bounding box coordinates if the image was resized for the detection

    Args:
        :detections (dict): The dictionary that outputs the model
        :min_score_thresh (float): The minimum score for the detections (detections with a score lower than this value will be discarded)
        :shape (tuple): The shape of the image
        :new_old_shape (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
                                the second element represents the bottom padding (applied by resize_preserving_ar() function) and
                                the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
                                the coordinates changes that we have to do)
            (default is None)

    Returns:
        :filtered_detections (dict): dictionary with detection scores, classes, centroids and bounding box coordinates ordered by score in descending order
    """
    allowed_categories = ["person"]
    # allowed_categories = ["Face"]  # if ssd face model

    im_height, im_width, _ = shape
    center_net = False

    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    key_points_score = None
    key_points = None

    if 'detection_keypoint_scores' in detections:
        key_points_score = detections['detection_keypoint_scores'][0].numpy()
        key_points = detections['detection_keypoints'][0].numpy()
        center_net = True

    sorted_index = np.argsort(scores)[::-1]
    scores = scores[sorted_index]
    boxes = boxes[sorted_index]
    classes = classes[sorted_index]

    i = 0
    while i < 10000:
        if scores[i] < min_score_thresh:  # sorted
            break
        if coco_category_index[classes[i]]["name"] in allowed_categories:
            i += 1
        else:
            scores = np.delete(scores, i)
            boxes = delete_items_from_array_aux(boxes, i)
            classes = np.delete(classes, i)
            if center_net:
                key_points_score = delete_items_from_array_aux(key_points_score, i)
                key_points = delete_items_from_array_aux(key_points, i)

    filtered_detections = dict()
    filtered_detections['detection_classes'] = classes[:i]

    rescaled_boxes = (boxes[:i])

    if new_old_shape:
        rescale_bb(rescaled_boxes, new_old_shape, im_width, im_height)
        if center_net:
            rescaled_key_points = key_points[:i]
            rescale_key_points(rescaled_key_points, new_old_shape, im_width, im_height)

    filtered_detections['detection_boxes'] = rescaled_boxes
    filtered_detections['detection_scores'] = scores[:i]

    if center_net:
        filtered_detections['detection_keypoint_scores'] = key_points_score[:i]
        filtered_detections['detection_keypoints'] = rescaled_key_points

    aux_centroids = []
    for bb in boxes[:i]:  # y_min, x_min, y_max, x_max
        centroid_x = (bb[1] + bb[3]) / 2.
        centroid_y = (bb[0] + bb[2]) / 2.
        aux_centroids.append([centroid_x, centroid_y])

    filtered_detections['detection_boxes_centroid'] = np.array(aux_centroids)

    return filtered_detections


# def detect_head_pose_ssd_face(image, detections, model, output_image):
#     """
#     Detect objects in the image running the model
#
#     Args:
#         :model (tensorflow.python.saved_model): The Tensorflow object detection model
#         :image (numpy.ndarray): The image that is given as input to the object detection model
#         :min_score_threshold (float): The minimum score for the detections (detections with a score lower than this value will be discarded)
#         :new_old_shape (tuple): The first element represents the right padding (applied by resize_preserving_ar() function);
#                                 the second element represents the bottom padding (applied by resize_preserving_ar() function) and
#                                 the third element is a tuple that is the shape of the image after resizing without the padding (this is useful for
#                                 the coordinates changes that we have to do)
#
#     Returns:
#         :detections (dict): dictionary with detection scores, classes, centroids and bounding box coordinates ordered by score in descending order
#         :inference_time (float): inference time for one image expressed in seconds
#     """
#
#     im_width, im_height = image.shape[1], image.shape[0]
#     classes = detections['detection_classes']
#     boxes = detections['detection_boxes']
#
#     i = 0
#     while i < len(classes):  # for each bb (person)
#         [y_min_perc, x_min_perc, y_max_perc, x_max_perc] = boxes[i]
#         (x_min, x_max, y_min, y_max) = (int(x_min_perc * im_width), int(x_max_perc * im_width), int(y_min_perc * im_height), int(y_max_perc * im_height))
#
#         y_min_face, x_min_face, y_max_face, x_max_face = enlarge_bb(y_min, x_min, y_max, x_max, im_width, im_height)
#         img_face = image[y_min_face:y_max_face, x_min_face:x_max_face]
#         img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
#
#         # img_face, _ = resize_preserving_ar(img_face, (224, 224))
#         img_face = cv2.resize(img_face, (224, 224))
#
#         img_face = np.expand_dims(img_face, axis=0)
#         yaw, pitch, roll = model.get_angle(img_face)
#
#         cv2.rectangle(output_image, (x_min_face, y_min_face), (x_max_face, y_max_face), (0, 0, 0), 2)
#         # cv2.imshow("aa", output_image)
#         # cv2.waitKey(0)
#         # to original image coordinates
#         x_min_orig, x_max_orig, y_min_orig, y_max_orig = x_min_face, x_max_face, y_min_face, y_max_face  # x_min_face + x_min, x_max_face + x_min, y_min_face + y_min, y_max_face+y_min
#         draw_axis(output_image, yaw, pitch, roll, tdx=(x_min_orig + x_max_orig) / 2, tdy=(y_min_orig + y_max_orig) / 2,
#                   size=abs(x_max_face - x_min_face))
#
#         i += 1
#
#     return output_image
#
#
# def detect_head_pose(image, detections, model, detector, output_image):
#     """
#     Detect the pose of the head given an image and the person detected
#
#     Args:
#         :image (numpy.ndarray): The image that is given as input
#         :detections (dict):  dictionary with detection scores, classes, centroids and bounding box coordinates ordered by score in descending order
#         :model (src.ai.whenet.WHENet): model to detect the pose of the head
#         :detector (_dlib_pybind11.cnn_face_detection_model_v1): model to detect the face
#         :output_image (numpy.ndarray): The output image where the drawings of the head pose will be done
#
#     Returns:
#         :output_image (numpy.ndarray): The output image with the drawings of the head pose
#     """
#
#     im_width, im_height = image.shape[1], image.shape[0]
#     classes = detections['detection_classes']
#     boxes = detections['detection_boxes']
#
#     i = 0
#     while i < len(classes):  # for each bb (person)
#         [y_min_perc, x_min_perc, y_max_perc, x_max_perc] = boxes[i]
#         (x_min, x_max, y_min, y_max) = (int(x_min_perc * im_width), int(x_max_perc * im_width), int(y_min_perc * im_height), int(y_max_perc * im_height))
#
#         img_person = image[y_min:y_max, x_min:x_max]
#
#         # start_time = time.time()
#         # img_face = img_person[:int(img_person.shape[0]/2), :]
#         rect_faces = detection_dlib_cnn_face(detector,  img_person)
#         # # rect_faces = detection_dlib_face(detector,  img_person)
#         # end_time = time.time()
#         # # print("Inference time dlib cnn: ", end_time - start_time)
#
#         if len(rect_faces) > 0:  # if the detector able to find faces
#
#             x_min_face, y_min_face, x_max_face, y_max_face = rect_faces[0][0], rect_faces[0][1], rect_faces[0][2], rect_faces[0][3]  # rect_faces[0][1]
#             y_min_face, x_min_face, y_max_face, x_max_face = enlarge_bb(y_min_face, x_min_face, y_max_face, x_max_face, im_width, im_height)
#
#             img_face = img_person[y_min_face:y_max_face, x_min_face:x_max_face]
#
#             img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
#
#             # img_face, _ = resize_preserving_ar(img_face, (224, 224))
#             img_face = cv2.resize(img_face, (224, 224))
#
#             img_face = np.expand_dims(img_face, axis=0)
#             # start_time = time.time()
#             yaw, pitch, roll = model.get_angle(img_face)
#             # end_time = time.time()
#             # print("Inference time whenet: ", end_time - start_time)
#
#             cv2.rectangle(output_image, (x_min_face + x_min, y_min_face + y_min), (x_max_face + x_min, y_max_face + y_min), (0, 0, 0), 2)
#             # to original image coordinates
#             x_min_orig, x_max_orig, y_min_orig, y_max_orig = x_min_face + x_min, x_max_face + x_min, y_min_face + y_min, y_max_face+y_min
#             draw_axis(output_image, yaw, pitch, roll, tdx=(x_min_orig + x_max_orig) / 2, tdy=(y_min_orig + y_max_orig) / 2,
#                       size=abs(x_max_face - x_min_face))
#             # draw_axis(image, yaw, pitch, roll, tdx=(x_min_face + x_max_face) / 2, tdy=(y_min_face + y_max_face) / 2,
#             #           size=abs(x_max_face - x_min_face))
#         else:  # otherwise
#             # print("SHAPE ", img_person.shape)
#             # x_min_face, y_min_face, x_max_face, y_max_face = int(img_person.shape[1]/8), 0, int(img_person.shape[1]-img_person.shape[1]/9), int(img_person.shape[0]/3)
#             # img_face = img_person[y_min_face:y_max_face, x_min_face:x_max_face]
#             # # img_face = resize_preserving_ar(img_face, (224, 224))
#             # img_face = cv2.resize(img_face, (224, 224))
#             # cv2.imshow("face_rsz", img_face)
#             # cv2.waitKey(0)
#             # img_face = np.expand_dims(img_face, axis=0)
#             # # cv2.rectangle(img_face, (x_min_face, y_min_face), (x_max_face, y_max_face), (0, 0, 0), 1)
#             # yaw, pitch, roll = model.get_angle(img_face)
#             # print("YPR", yaw, pitch, roll)
#             # draw_axis(img_person, yaw, pitch, roll, tdx=(x_min_face+x_max_face)/2, tdy=(y_min_face+y_max_face)/2, size=abs(x_max_face-x_min_face))
#             # cv2.imshow('output', img_person)
#             # cv2.waitKey(0)
#             i += 1
#             continue
#
#         i += 1
#
#     return output_image


# def detect_head_pose_whenet(model, person, image):
#
#     """
#     Detect the head pose using the whenet model and draw on image
#
#     Args:
#         :model (): Whenet model
#         :person ():
#         :image (numpy.ndarray): The image that is given as input
#
#     Returns:
#         :
#     """
#
#     faces_coordinates = person.get_faces_coordinates()[-1]
#
#     y_min, x_min, y_max, x_max = faces_coordinates
#
#     image_face = image[y_min:y_max, x_min:x_max]
#     img_face = cv2.cvtColor(image_face, cv2.COLOR_BGR2RGB)
#
#     # img_face, _ = resize_preserving_ar(img_face, (224, 224))
#     img_face = cv2.resize(img_face, (224, 224))
#
#     img_face = np.expand_dims(img_face, axis=0)
#     # start_time = time.time()
#     yaw, pitch, roll = model.get_angle(img_face)
#
#     # end_time = tiypme.time()
#     # print("Inference time whenet: ", end_time - start_time)
#     # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
#
#     # to original image coordinates
#     x_min_orig, x_max_orig, y_min_orig, y_max_orig = x_min, x_max, y_min, y_max
#     vector_norm = draw_axis(image, yaw, pitch, roll, tdx=(x_min_orig + x_max_orig) / 2, tdy=(y_min_orig + y_max_orig) / 2,
#               size=abs(x_max - x_min))
#
#
#     visualize_vector(image, [int((x_min_orig + x_max_orig) / 2), int((y_min_orig + y_max_orig) / 2)], vector_norm)
#
#     person.update_poses_ypr([yaw, pitch, roll])
#     person.update_poses_vector_norm(vector_norm)

    # cv2.imshow("", image)
    # cv2.waitKey(0)
