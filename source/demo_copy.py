import argparse

import cv2
import sys
import numpy as np

def initialize_zed_camera(input_file=None):
    print('initialize zed camera')
    print('input_file = ', input_file)
    return 0, 0

def load_image():
    pass


def infer_ypr():
    pass

def extract_keypoints_zedcam(zed):
     print('extract keypoints zedcam')



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




if __name__=="__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=None, help="path to the model", required=True)
    ap.add_argument("-f", "--input-file", type=str, default=None, help="input a SVO file", required=False)
    config = ap.parse_args()

    print(config.input_file)
    print(config.model)

    zed, run_parameters = initialize_zed_camera(input_file=config.input_file)

    if str(config.model).lower() == 'zed':
        extract_keypoints_zedcam(zed=zed) # everything performed with stereilabs SDK
    elif str(config.model).lower() == 'centernet':
        print('centernet')
        raise NotImplementedError
    elif str(config.model).lower()=='openpose':
        print('openpose')
        raise NotImplementedError
    else:
        print('wrong input for model value')
        raise IOError # probably not correct error


