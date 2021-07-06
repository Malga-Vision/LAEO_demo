import math
import os
import numpy as np
import tensorflow as tf

from utils.my_utils import normalize_wrt_maximum_distance_point, retrieve_interest_points


def head_pose_estimation(kpt, detector, gaze_model, id_list=None):
    fps, shape = 20, (1280, 720)

    yaw_list, pitch_list, roll_list, yaw_u_list, pitch_u_list, roll_u_list = [], [], [], [], [], []
    center_xy = []

    for j, kpt_person in enumerate(kpt):
        # TODO here change order if openpose
        face_kpt = retrieve_interest_points(kpt_person, detector=detector)

        tdx = np.mean([face_kpt[k] for k in range(0, 15, 3) if face_kpt[k] != 0.0])
        tdy = np.mean([face_kpt[k + 1] for k in range(0, 15, 3) if face_kpt[k + 1] != 0.0])
        if math.isnan(tdx) or math.isnan(tdy):
            tdx = -1
            tdy = -1

        center_xy.append([tdx, tdy])
        face_kpt_normalized = np.array(normalize_wrt_maximum_distance_point(face_kpt))
        # print(type(face_kpt_normalized), face_kpt_normalized)

        aux = tf.cast(np.expand_dims(face_kpt_normalized, 0), tf.float32)

        yaw, pitch, roll = gaze_model(aux, training=False)
        # print(yaw[0].numpy()[0], pitch, roll)
        yaw_list.append(yaw[0].numpy()[0])
        pitch_list.append(pitch[0].numpy()[0])
        roll_list.append(roll[0].numpy()[0])

        yaw_u_list.append(yaw[0].numpy()[1])
        pitch_u_list.append(pitch[0].numpy()[1])
        roll_u_list.append(roll[0].numpy()[1])
        # print(id_lists[j])
        # print('yaw: ', yaw[0].numpy()[0], 'yaw unc: ', yaw[0].numpy()[1], 'pitch: ', pitch[0].numpy()[0],
        #       'pitch unc: ', pitch[0].numpy()[1], 'roll: ', roll[0].numpy()[0], 'roll unc: ', roll[0].numpy()[1])
        # draw_axis(yaw.numpy(), pitch.numpy(), roll.numpy(), im_pose, tdx, tdy)
    return center_xy, yaw_list, pitch_list, roll_list

def hpe(gaze_model, kpt_person, detector):
    # TODO here change order if openpose
    face_kpt = retrieve_interest_points(kpt_person, detector=detector)

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

    return yaw, pitch, roll, tdx, tdy

def project_ypr_in2d(yaw, pitch, roll):
    """ Project yaw pitch roll on image plane. Result is NOT normalised.

    :param yaw:
    :param pitch:
    :param roll:
    :return:
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    x3 = (math.sin(yaw))
    y3 = (-math.cos(yaw) * math.sin(pitch))

    # normalize the components
    length = np.sqrt(x3**2 + y3**2)

    return [x3, y3]


