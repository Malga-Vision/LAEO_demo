import argparse
import os
import numpy as np
import tensorflow as tf
import cv2
from intialize_camera import initialize_zed_camera
import pyzed.sl as sl
from source.utils.img_util import resize_preserving_ar, draw_detections, percentage_to_pixel
from source.utils.my_utils import retrieve_xyz_from_detection, compute_distance
from source.ai.detection import detect


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=None, help="path to the model", required=True)
    config = ap.parse_args()

    tf.keras.backend.clear_session()
    model = tf.saved_model.load(os.path.join(config.model, 'saved_model'))
    input_shape_od_model = (512, 512)

    zed, runtime_parameters = initialize_zed_camera()
    image, depth_image, point_cloud = sl.Mat(), sl.Mat(), sl.Mat()

    # params
    min_score_thresh, max_boxes_to_draw, min_distance = .45, 50, 1.5
    i = 15
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  # A new image is available if grab() returns SUCCESS

            zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve left image
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # Colored point cloud. Each pixel contains 4 float (X, Y, Z, color)

            pc_img = point_cloud.get_data()[:, :, :3]
            img = np.array(image.get_data()[:, :, :3])

            img, new_old_shape = resize_preserving_ar(img, input_shape_od_model)
            detections, elapsed_time = detect(model, img, min_score_thresh, new_old_shape)  # detection classes boxes scores
            img_with_detections = draw_detections(img, detections, max_boxes_to_draw, None, None, None)

            det, kpt = percentage_to_pixel(img.shape, detections['detection_boxes'], detections['detection_scores'],
                                           detections['detection_keypoints'], detections['detection_keypoint_scores'])

            XYZ = retrieve_xyz_from_detection(detections['detection_boxes_centroid'], pc_img)
            _, violate, couple_points = compute_distance(XYZ, min_distance)
            img_with_violations = draw_detections(img, detections, max_boxes_to_draw, violate, couple_points)

            cv2.imshow("aa", img_with_violations)
            cv2.waitKey(0)
            i += 1

    zed.close()  # Close the camera




