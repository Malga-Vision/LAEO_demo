"""
   This sample shows how to detect a human bodies and draw their
   modelised skeleton in an window
   Calculate the direction of view form face keypoints and draw the view line.
   The line is coloured: GREEN -> high ocular interaction, BLACk -> low interaction
"""
import argparse
import os

import cv2
import numpy as np
import pyzed.sl as sl
import tensorflow as tf


from ai.detection import detect
from ai.tracker import Sort
from laeo_per_frame.interaction_per_frame_uncertainty import LAEO_computation
from utils.hpe import hpe, project_ypr_in2d
from utils.img_util import resize_preserving_ar, draw_detections, percentage_to_pixel, draw_key_points_pose, \
    visualize_vector, draw_key_points_pose_zedcam

# from utils.my_utils import retrieve_xyz_from_detection, compute_distance, save_key_points_to_json
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
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_MEDIUM  # HUMAN_BODY_FAST, HUMAN_BODY_MEDIUM ->less fast but more accurate

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
#../models/hpe_model/bhp-net_model
    gaze_model = tf.keras.models.load_model('../models', custom_objects={"tf": tf})

    # Grab an image
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image, image is unsigned char of 4 channels
        # if len(tf.config.list_physical_devices('GPU'))>0:
        #     zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.GPU, display_resolution)
        # else:
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        # Retrieve objects
        zed.retrieve_objects(bodies, obj_runtime_param)

        # image_left_ocv = image.get_data()
        image_left_ocv = np.array(image.get_data()[:, :, :3])
        image_left_ocv = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGR2GRAY)
        image_left_ocv = cv2.cvtColor(image_left_ocv, cv2.COLOR_GRAY2RGB)

        people_list =[]
        kpt = 0
        # for j, obj_person in enumerate(bodies.object_list):
        #     kpt = np.asarray(obj_person.keypoint_2d)
        for j, obj_person in enumerate(bodies.object_list):
            # kpt = np.asarray(obj_person.keypoint_2d)
            confidence = np.asarray(obj_person.keypoint_confidence)
            # replace nana with 0 (zero)
            where_are_NaNs = np.isnan(confidence)
            confidence[where_are_NaNs] = 0
            kpt = np.c_[obj_person.keypoint_2d, confidence/99] # 99 is the max

            yaw, pitch, roll, tdx, tdy = hpe(gaze_model, kpt, detector='zedcam')

            # img = draw_axis_3d(yaw[0].numpy()[0], pitch[0].numpy()[0], roll[0].numpy()[0], image=img, tdx=tdx, tdy=tdy,
            #                    size=50)

            people_list.append({'yaw': yaw[0].numpy()[0],
                                'yaw_u': 0,
                                'pitch': pitch[0].numpy()[0],
                                'pitch_u': 0,
                                'roll': roll[0].numpy()[0],
                                'roll_u': 0,
                                'center_xy': [tdx, tdy]})


        for obj_person in bodies.object_list:
            image_left_ocv = draw_key_points_pose_zedcam(image_left_ocv, obj_person.keypoint_2d)

        # cv2.imshow("ZED | 2D View", image_left_ocv)

        # call LAEO
        clip_uncertainty = 0.5
        binarize_uncertainty = False
        interaction_matrix = LAEO_computation(people_list, clipping_value=clip_uncertainty, clip=binarize_uncertainty)
        # coloured arrow print per person
        #TODO coloured arrow print per person
        for index, person in enumerate(people_list):
            green = round((max(interaction_matrix[index, :])) * 255)
            colour = (0, green, 0)
            if green < 40:
                colour =(0,0,255)
            vector = project_ypr_in2d(person['yaw'], person['pitch'], person['roll'])
            image_left_ocv = visualize_vector(image_left_ocv, person['center_xy'], vector, title="", color=colour)
        cv2.namedWindow('MaLGa Lab Demo', cv2.WINDOW_NORMAL)
        cv2.imshow('MaLGa Lab Demo', image_left_ocv)
        # cv2.resizeWindow('MaLGa Lab Demo', 400, 400)
        try:
            laeo_1, laeo_2 = (np.unravel_index(np.argmax(interaction_matrix, axis=None), interaction_matrix.shape))
            # print something around face
        except:
            pass

        # img = np.array(image.get_data()[:, :, :3])
        #
        # cv2.imshow("ZED | 2D View", img)

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


def extract_keypoints_centernet(path_to_model, zed, gaze_model_path):
    """

    :param model:
    :param zed:
    """

    model = tf.saved_model.load(os.path.join(path_to_model, 'saved_model'))
    # model = tf.keras.models.load_model(gaze_model_path, custom_objects={"tf": tf})

    input_shape_od_model = (512, 512)

    image, image_cpu, depth_image, point_cloud = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()

    # params
    min_score_thresh, max_boxes_to_draw, min_distance = .45, 50, 1.5

    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    # call HPE
    print('load hpe')
    gaze_model = tf.keras.models.load_model(gaze_model_path, custom_objects={"tf": tf})

    # tracker stuff
    mot_tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.4)

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        print('image retrieved')
        # if tf.test.is_gpu_available():
        #     zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.GPU, display_resolution)
        #     zed.retrieve_image(image_cpu, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        # else:
        zed.retrieve_image(image_cpu, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # image_cpu = image.get_pointer()
        # Retrieve objects
        # zed.retrieve_objects(bodies, obj_runtime_param)

        img = np.array(image_cpu.get_data()[:, :, :3])

        img_resized, new_old_shape = resize_preserving_ar(img, input_shape_od_model)

        print('inference centernet')
        detections, elapsed_time = detect(model, img_resized, min_score_thresh,
                                          new_old_shape)  # detection classes boxes scores
        # probably to draw on resized
        img_with_detections = draw_detections(img_resized, detections, max_boxes_to_draw, None, None, None)
        # cv2.imshow("aa", img_with_detections)

        det, kpt = percentage_to_pixel(img.shape, detections['detection_boxes'], detections['detection_scores'],
                                       detections['detection_keypoints'], detections['detection_keypoint_scores'])

        # tracker stuff
        trackers = mot_tracker.update(det, kpt)
        people = mot_tracker.get_trackers()

        # center_xy, yaw, pitch, roll = head_pose_estimation(kpt, 'centernet', gaze_model=gaze_model)

        # _________ extract hpe and print to img
        people_list = []
        vector_list = []

        print('inferece hpe')

        for j, kpt_person in enumerate(kpt):
            yaw, pitch, roll, tdx, tdy = hpe(gaze_model, kpt_person, detector='centernet')

            # img = draw_axis_3d(yaw[0].numpy()[0], pitch[0].numpy()[0], roll[0].numpy()[0], image=img, tdx=tdx, tdy=tdy,
            #                    size=50)

            people_list.append({'yaw': yaw[0].numpy()[0],
                                'yaw_u': 0,
                                'pitch': pitch[0].numpy()[0],
                                'pitch_u': 0,
                                'roll': roll[0].numpy()[0],
                                'roll_u': 0,
                                'center_xy': [tdx, tdy]})

        for i in range(len(det)):
            img = draw_key_points_pose(img, kpt[i])
        # img = draw_axis(yaw[i], pitch[i], roll[i], image=img, tdx=center_xy[0], tdy=center_xy[1], size=50) #single person

        # call LAEO
        clip_uncertainty = 0.5
        binarize_uncertainty = False
        interaction_matrix = LAEO_computation(people_list, clipping_value=clip_uncertainty, clip=binarize_uncertainty)
        # coloured arrow print per person
        #TODO coloured arrow print per person

        print('before cv2')

        for index, person in enumerate(people_list):
            green = round((max(interaction_matrix[index, :])) * 255)
            colour = (0, green, 0)
            vector = project_ypr_in2d(person['yaw'], person['pitch'], person['roll'])
            img = visualize_vector(img, person['center_xy'], vector, title="", color=colour)
        cv2.namedWindow('MaLGa Lab Demo', cv2.WINDOW_NORMAL)
        cv2.imshow('MaLGa Lab Demo', img)
        # cv2.resizeWindow('MaLGa Lab Demo', 400, 400)

        print('after cv2')

        try:
            laeo_1, laeo_2 = (np.unravel_index(np.argmax(interaction_matrix, axis=None), interaction_matrix.shape))
            # print something around face
        except:
            pass
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # save_key_points_to_json(kpts, path_json + ".json")

        # XYZ = retrieve_xyz_from_detection(detections['detection_boxes_centroid'], pc_img)
        # _, violate, couple_points = compute_distance(XYZ, min_distance)
        # img_with_violations = draw_detections(img, detections, max_boxes_to_draw, violate, couple_points)

    print('image free GPU/CPU')
    image_cpu.free(sl.MEM.CPU)
    # image.free(sl.MEM.GPU)
    cv2.destroyAllWindows()

    # Disable modules and close camera
    zed.close()


if __name__ == "__main__":
    """Example of usage:
            -m zed
            [-f /media/DATA/Users/Federico/Zed_Images/HD720_SN24782978_14-06-59.svo]
        or 
            -m centernet
            [-f /your_file]
        m: identifies the keypoints extractor algorithm
        f: a pre-recorded zedcam file, .svo format"""

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=None, help="path to the model", required=True)
    ap.add_argument("-f", "--input-file", type=str, default=None, help="input a SVO file", required=False)
    config = ap.parse_args()

    # choose between real time and pre-recorded file
    if config.input_file is not None:
        print('video file {}'.format(config.input_file))
    else:
        print('real time camera acquisition')

    # initialize zedcam with the proper function
    zed, run_parameters = initialize_zed_camera(input_file=config.input_file)

    # choose the keypoints extractor algorithm and run it on every frame, here the program remains until finished
    if str(config.model).lower() == 'zed':
        print('start zedcam keypoint extractor')
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        extract_keypoints_zedcam(zed=zed)  # everything performed with stereolabs SDK
    elif str(config.model).lower() == 'centernet':
        print('start centernet keypoint extractor')
        # path_to_model = '/media/DATA/Users/Federico/centernet_hg104_512x512_kpts_coco17_tpu-32'
        # path to your centernet model: https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1
        path_to_model = '/home/federico/Documents/Models_trained/keypoint_detector/centernet_hg104_512x512_kpts_coco17_tpu-32'
        if not os.path.isdir(path_to_model):
            path_to_model = '/home/federico/Documents/Models_trained/keypoint_detector/centernet_hg104_512x512_kpts_coco17_tpu-32'
            if not os.path.isdir(path_to_model):
                raise IOError('path for model is incorrect, cannot find centernet model')

        # tf.keras.backend.clear_session()

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # gpus = tf.config.list_physical_devices('GPU')

        # with tf.device(gpus[0]):

        # path_to_model = tf.compat.v1.saved_model.load(os.path.join(path_to_model, 'saved_model'))
        print('start centernet')
        gaze_model_path = '/home/federico/Documents/Models_trained/head_pose_estimation'
        extract_keypoints_centernet(path_to_model, zed, gaze_model_path=gaze_model_path)
    elif str(config.model).lower() == 'openpose':
        print('openpose')
        raise NotImplementedError
    else:
        print('wrong input for model value')
        raise IOError('wrong model name. Try with \'zed\' or \'centernet\' ') # probably not correct error