"""
   This sample shows how to detect a human bodies and draw their
   modelised skeleton in an window
"""
import argparse
import math
import os

import cv2
import numpy as np
import pyzed.sl as sl
import tensorflow as tf
from laeo_per_frame.interaction_per_frame_uncertainty import LAEO_computation
from utils.hpe import head_pose_estimation, hpe, project_ypr_in2d
from utils.img_util import resize_preserving_ar, draw_detections, percentage_to_pixel, draw_key_points_pose, \
    draw_axis, draw_axis_3d, visualize_vector, draw_key_points_pose_zedcam
from ai.tracker import Sort


# from utils.my_utils import retrieve_xyz_from_detection, compute_distance, save_key_points_to_json
from ai.detection import detect
# from utils.my_utils import normalize_wrt_maximum_distance_point, retrieve_interest_points

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
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # depth mode (NONE/PERFORMANCE/QUALITY/ULTRA)
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
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # sensing mode (STANDARD/FILL)
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100 # 100 -> no pixel is rejected
    runtime_parameters.texture_confidence_threshold = 100

    # mirror_ref = sl.Transform()
    # mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
    # tr_np = mirror_ref.m

    return zed, runtime_parameters