'''It calculates interaction frame per frame with not temporal consistency'''
import json
import jsonpickle
import numpy as np

from src.load_dataset_files.utils_load_predictions import load_head_3d_file
from pathlib import Path
from src.threeD_src.interaction_3d import Interaction_3d, project_ypr_in2d
from src.load_dataset_files.utils_load_predictions import bbox_creator

import re


def calculate_uncertainty(yaw_1, pitch_1, roll_1, clipping_value, clip=True):
    # res_1 = abs((pitch_1 + yaw_1 + roll_1) / 3)
    res_1 = abs((pitch_1 + yaw_1) / 2)
    if clip:
        # it binarize the uncertainty
        if res_1 > clipping_value:
            res_1 = clipping_value
        else:
            res_1 = 0
    else:
        # it leaves uncertainty untouched except for upper bound
        if res_1 > clipping_value:
            res_1 = clipping_value
        elif res_1 < 0:
            res_1 = 0

    # normalize
    res_1 = res_1 / clipping_value
    # assert res_1 in [0, 1], 'uncertainty not binarized'
    return res_1


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
           alist.sort(key=natural_keys) sorts in human order
           http://nedbatchelder.com/blog/200712/human_sorting.html
           (See Toothy's implementation in the comments)
           '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def delete_file_if_exist(*file_path):
    for f in file_path:
        if f.is_file():  # if exist already, replace
            f.unlink(missing_ok=True)


def LAEO_computation(people_list, clipping_value, clip):
    people_in_frame = len(people_list)

    # create empty matrix with one entry per person in frame
    matrix = np.empty((people_in_frame, people_in_frame))
    interaction_matrix = np.zeros((people_in_frame, people_in_frame))
    uncertainty_matrix = np.zeros((people_in_frame, people_in_frame))

    norm_xy_all = []  # it will contains vector for printing
    for subject in range(people_in_frame):
        norm_xy = project_ypr_in2d(people_list[subject]['yaw'], people_list[subject]['pitch'],
                                   people_list[subject]['roll'])
        norm_xy_all.append(norm_xy)
        for object in range(people_in_frame):
            v = Interaction_3d._compute_interaction_cosine(people_list[subject]['center_xy'], norm_xy,
                                                           people_list[object]['center_xy'])
            matrix[subject][object] = v

    # matrix is completed

    for subject in range(people_in_frame):
        uncertainty_1 = calculate_uncertainty(people_list[subject]['yaw_u'],
                                              people_list[subject]['pitch_u'],
                                              people_list[subject]['roll_u'], clipping_value=clipping_value,
                                              clip=clip)
        for object in range(people_in_frame):  # could be computed once, stored and then multiplied in a for(for())
            uncertainty_2 = calculate_uncertainty(people_list[object]['yaw_u'],
                                                  people_list[object]['pitch_u'],
                                                  people_list[object]['roll_u'], clipping_value=clipping_value,
                                                  clip=clip)
            # take average of previous matrix
            v = (matrix[subject][object] + matrix[object][subject]) / 2
            interaction_matrix[subject][object] = v * (((1 - uncertainty_1) + (1 - uncertainty_2)) / 2)
            uncertainty_matrix[subject][object] = uncertainty_1
            # uncertainty_matrix[object][subject] = uncertainty_2

    # couple with max LAEO
    laeo_1, laeo_2 = (np.unravel_index(np.argmax(interaction_matrix, axis=None), interaction_matrix.shape))

    return interaction_matrix


if __name__ == '__main__':
    clip_uncertainty = 0
    binarize_uncertainty = True
    yaw, pitch, roll, tdx, tdy = 0, 0, 0, 0, 0
    my_list = [{'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'center_xy': [tdx, tdy]}]
    _ = LAEO_computation(my_list, clipping_value=clip_uncertainty, clip=binarize_uncertainty)
