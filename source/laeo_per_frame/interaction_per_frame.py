'''It calculates interaction frame per frame with not temporal consistency'''
import re

import numpy as np

from laeo_per_frame.interaction_per_frame_uncertainty import project_ypr_in2d

def _compute_interaction_cosine(head_position, gaze_direction, target, visual_cone=True):
    """Computes the interaction between two people using the angle of view.

    The interaction in measured as the cosine of the angle formed by the line from person A to B
    and the gaze direction of person A.
    Reference system of zero degree:


    :param head_position: position of the head of person A
    :param gaze_direction: gaze direction of the head of person A
    :param target: position of head of person B
    :param yaw:
    :param pitch:
    :param roll:
    :param visual_cone: (default) True, if False gaze is a line, otherwise it is a cone (more like humans)
    :return: float or double describing the quantity of interaction
    """
    if np.array_equal(head_position, target):
        return 0  # or -1
    else:
        # direction from observer to target
        _direction_ = np.arctan2((target[1] - head_position[1]), (target[0] - head_position[0]))
        _direction_gaze_ = np.arctan2(gaze_direction[1], gaze_direction[0])
        difference = _direction_ - _direction_gaze_  # radians
        # TODO implemnt visual_cone as inverse of the 'uncertanty' parameter
        # TODO implement difference with and without visual_cone
        if visual_cone and (0 < difference < np.deg2rad(5)):
            difference = 0
        # difference of the line joining observer -> target with the gazing direction,

        val = np.cos(difference)
        if val < 0:
            return 0
        else:
            return val

def _compute_interaction_cosine(head_position, gaze_direction, target, visual_cone=True):
    """Computes the interaction between two people using the angle of view.

    The interaction in measured as the cosine of the angle formed by the line from person A to B
    and the gaze direction of person A.
    Reference system of zero degree:


    :param head_position: position of the head of person A
    :param gaze_direction: gaze direction of the head of person A
    :param target: position of head of person B
    :param yaw:
    :param pitch:
    :param roll:
    :param visual_cone: (default) True, if False gaze is a line, otherwise it is a cone (more like humans)
    :return: float or double describing the quantity of interaction
    """
    if np.array_equal(head_position, target):
        return 0  # or -1
    else:
        # direction from observer to target
        _direction_ = np.arctan2((target[1] - head_position[1]), (target[0] - head_position[0]))
        _direction_gaze_ = np.arctan2(gaze_direction[1], gaze_direction[0])
        difference = _direction_ - _direction_gaze_  # radians
        # TODO implemnt visual_cone as inverse of the 'uncertanty' parameter
        # TODO implement difference with and without visual_cone
        if visual_cone and (0 < difference < np.deg2rad(5)):
            difference = 0
        # difference of the line joining observer -> target with the gazing direction,

        val = np.cos(difference)
        if val < 0:
            return 0
        else:
            return val
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
            v = _compute_interaction_cosine(people_list[subject]['center_xy'], norm_xy,
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
    clip_uncertainty = 0.1
    binarize_uncertainty = True
    yaw, pitch, roll, tdx, tdy = 0, 0, 0, 0, 0
    my_list = [{'yaw': yaw,
                'yaw_u': 0,
                'pitch': pitch,
                'pitch_u': 0,
                'roll': roll,
                'roll_u': 0,
                'center_xy': [tdx, tdy]}]
    _ = LAEO_computation(my_list, clipping_value=clip_uncertainty, clip=binarize_uncertainty)
