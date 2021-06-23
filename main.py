import tensorflow as tf
import cv2
import os
import math
import numpy as np
from pathlib import Path
from my_utils import normalize_wrt_maximum_distance_point, initialize_video_recorder, retrieve_interest_points, load_poses_from_json, load_poses_from_json1, ypr_to_json


def load_image():
    pass


def infer_ypr():
    pass

def extract_keypoints():
    pass


def compute_laeo():
    pass


def save_files():
    pass


if __name__ == '__main__':

    # argparse or zedcam input

    load_image()
    extract_keypoints()
    infer_ypr()
    compute_laeo()

    save_files()
