"""
   This sample shows how to detect a human bodies and draw their
   modelised skeleton in an OpenGL window
"""
import argparse
import math
import os

import cv2
import numpy as np
import pyzed.sl as sl
import tensorflow as tf
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
from source.utils.hpe import head_pose_estimation
from source.utils.img_util import resize_preserving_ar, draw_detections, percentage_to_pixel, draw_key_points_pose, \
    draw_axis, draw_axis_3d
from source.utils.my_utils import retrieve_xyz_from_detection, compute_distance, save_key_points_to_json
from source.ai.detection import detect
from source.utils.my_utils import normalize_wrt_maximum_distance_point, retrieve_interest_points


def initialize_zed_camera(input_file=None):
    """Create a Camera object, set the configurations parameters and open the camera

    Returns:
        :param input_file:
        :return:
        :zed (pyzed.sl.Camera): Camera object
    """
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY  # depth mode (PERFORMANCE/QUALITY/ULTRA)
    init_params.coordinate_units = sl.UNIT.METER  # depth measurements (METER/CENTIMETER/MILLIMETER/FOOT/INCH)
    init_params.camera_resolution = sl.RESOLUTION.HD720  # resolution (HD720/HD1080/HD2K)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    init_params.depth_minimum_distance = 0.40  # cm
    init_params.depth_maximum_distance = 15
    # init_params.depth_stabilization = False  # to improve computational performance

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if input_file is not None:
        filepath = input_file
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print('zedcam has problems')
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # sensing mode (STANDARD/FILL)
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # mirror_ref = sl.Transform()
    # mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
    # tr_np = mirror_ref.m

    return zed, runtime_parameters


def load_image():
    pass


def infer_ypr():
    pass


def extract_keypoints_zedcam(zed):
    """Zed Cam starts and extracts keypoints with stereolabs SDK object detector.

    :param zed: (pyzed.sl.Camera): Camera object
    """
    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True  # Smooth skeleton move
    obj_param.enable_tracking = True  # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST  # HUMAN_BODY_MEDIUM ->less fast but more accurate

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()

    # Grab an image
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image, image is unsigned char of 4 channels
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        # Retrieve objects
        zed.retrieve_objects(bodies, obj_runtime_param)

        # here we have bodies, extract and print

        img = np.array(image.get_data()[:, :, :3])

        cv2.imshow("ZED | 2D View", img)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image.free(sl.MEM.CPU)
    cv2.destroyAllWindows()

    # Disable modules and close camera
    zed.close()

    # with open('report_file.txt', 'w') as f:
    #     f.write(str(keypoints))
    #     f.write('\n')
    #     f.write(str(confidence))
    #     f.write('\n')
    #     f.write(str(bbox_3d))


def compute_laeo():
    pass


def save_files():
    pass


def myfunct():
    pass


# https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1BODY__PARTS.html
# NOSE
# NECK
# RIGHT_SHOULDER
# RIGHT_ELBOW
# RIGHT_WRIST
# LEFT_SHOULDER
# LEFT_ELBOW
# LEFT_WRIST
# RIGHT_HIP
# RIGHT_KNEE
# RIGHT_ANKLE
# LEFT_HIP
# LEFT_KNEE
# LEFT_ANKLE
# RIGHT_EYE
# LEFT_EYE
# RIGHT_EAR
# LEFT_EAR


def extract_keypoints_centernet(model, zed):
    input_shape_od_model = (512, 512)

    image, depth_image, point_cloud = sl.Mat(), sl.Mat(), sl.Mat()

    # params
    min_score_thresh, max_boxes_to_draw, min_distance = .45, 50, 1.5

    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        # Retrieve objects
        # zed.retrieve_objects(bodies, obj_runtime_param)

        img = np.array(image.get_data()[:, :, :3])

        img_resized, new_old_shape = resize_preserving_ar(img, input_shape_od_model)
        detections, elapsed_time = detect(model, img_resized, min_score_thresh, new_old_shape)  # detection classes boxes scores
        # probably to draw on resized
        img_with_detections = draw_detections(img_resized, detections, max_boxes_to_draw, None, None, None)
        # cv2.imshow("aa", img_with_detections)

        det, kpt = percentage_to_pixel(img.shape, detections['detection_boxes'], detections['detection_scores'],
                                       detections['detection_keypoints'], detections['detection_keypoint_scores'])

        # call HPE
        gaze_model = tf.keras.models.load_model('models/hpe_model/bhp-net_model', custom_objects={"tf": tf})
        # center_xy, yaw, pitch, roll = head_pose_estimation(kpt, 'centernet', gaze_model=gaze_model)

        for j, kpt_person in enumerate(kpt):
            # TODO here change order if openpose
            face_kpt = retrieve_interest_points(kpt_person, detector='centernet')

            tdx = np.mean([face_kpt[k] for k in range(0, 15, 3) if face_kpt[k] != 0.0])
            tdy = np.mean([face_kpt[k + 1] for k in range(0, 15, 3) if face_kpt[k + 1] != 0.0])
            if math.isnan(tdx) or math.isnan(tdy):
                tdx = -1
                tdy = -1

            # center_xy.append([tdx, tdy])
            face_kpt_normalized = np.array(normalize_wrt_maximum_distance_point(face_kpt))
            # print(type(face_kpt_normalized), face_kpt_normalized)

            aux = tf.cast(np.expand_dims(face_kpt_normalized, 0), tf.float32)

            yaw, pitch, roll = gaze_model(aux, training=False)

            print('yaw = {}'.format(yaw))
            print('yaw[0].numpy()[0] = {}'.format(yaw[0].numpy()[0]))
            print(tdx)

            img = draw_axis_3d(yaw[0].numpy()[0], pitch[0].numpy()[0], roll[0].numpy()[0], image=img, tdx=tdx, tdy=tdy, size=50)

        # call LAEO

        for i in range(len(det)):
            # img = draw_key_points_pose(img, kpt[i])
            try:
                print(yaw)
                print(np.shape(yaw))

            except:
                pass
            # img = draw_axis(yaw[i], pitch[i], roll[i], image=img, tdx=center_xy[0], tdy=center_xy[1], size=50) #single person

        print(type(img))
        cv2.imshow('bb', img)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # save_key_points_to_json(kpts, path_json + ".json")

        # XYZ = retrieve_xyz_from_detection(detections['detection_boxes_centroid'], pc_img)
        # _, violate, couple_points = compute_distance(XYZ, min_distance)
        # img_with_violations = draw_detections(img, detections, max_boxes_to_draw, violate, couple_points)

    image.free(sl.MEM.CPU)
    cv2.destroyAllWindows()

    # Disable modules and close camera
    zed.close()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=None, help="path to the model", required=True)
    ap.add_argument("-f", "--input-file", type=str, default=None, help="input a SVO file", required=False)
    config = ap.parse_args()

    if config.input_file is not None:
        print('video file {}'.format(config.input_file))
    else:
        print('real time camera acquisition')
    zed, run_parameters = initialize_zed_camera(input_file=config.input_file)

    if str(config.model).lower() == 'zed':
        print('start zedcam keypoint extractor')
        extract_keypoints_zedcam(zed=zed)  # everything performed with stereolabs SDK
    elif str(config.model).lower() == 'centernet':
        print('centernet')
        path_to_model = '/media/DATA/Users/Federico/centernet_hg104_512x512_kpts_coco17_tpu-32'
        tf.keras.backend.clear_session()
        print('siamo qui')
        path_to_model = tf.saved_model.load(os.path.join(path_to_model, 'saved_model'))
        print('start centernet')
        extract_keypoints_centernet(path_to_model, zed)
    elif str(config.model).lower() == 'openpose':
        print('openpose')
        raise NotImplementedError
    else:
        print('wrong input for model value')
        raise IOError  # probably not correct error
