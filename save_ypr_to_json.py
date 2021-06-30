import argparse
import tensorflow as tf
import cv2
import os
import math
import numpy as np
from source.utils.img_util import draw_key_points_pose, draw_axis, draw_cones
from source.utils.my_utils import normalize_wrt_maximum_distance_point, initialize_video_recorder, retrieve_interest_points, load_poses_from_json, load_poses_from_json1, ypr_to_json
from pathlib import Path

if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-hm", "--hpe-model", type=str, default=None, help="path to the hpe model", required=True)
    # ap.add_argument("-o", "--output", type=str, default=None, help="path to the video will be savoutputed", required=True)
    # config = ap.parse_args()

    fps, shape = 20, (1280, 720)
    # writer, _ = initialize_video_recorder(os.path.join(config.output, 'MR12.mp4.avi'), None, fps, shape)

    gaze_model = tf.keras.models.load_model('models/hpe_model/bhp-net_model', custom_objects={"tf": tf})

    path_folder = '/media/federico/HHD FEDE/ucolaeo-processed/v1.1/open_pose/untracked'
    img_folder = '/media/federico/HHD FEDE/ucolaeo-processed/v1.1/open_pose/tracked'
    keypoints_folder = '/media/federico/HHD FEDE/ucolaeo-processed/v1.1/open_pose/tracked'
    result_folder = '/media/federico/HHD FEDE/ucolaeo-processed/v1.1/open_pose/trial'

    openpose = True
    if openpose:
        detector = 'openpose'
    else:
        detector = 'centernet'

    for fold in os.listdir(path_folder):

        print(fold)

        path_ = os.path.join(path_folder, fold)
        t = os.path.join(img_folder, fold)
        path_imgs = os.path.join(t, 'images')
        t= os.path.join(keypoints_folder, fold)
        path_jsons = os.path.join(t, 'keypoints')
        t = os.path.join(result_folder, fold)

        if not os.path.exists(os.path.join(t, 'ypr')):
            os.makedirs(os.path.join(t, 'ypr'))

        for i in range(1, len(os.listdir(path_imgs))):
            # print(i)
            if Path(os.path.join(path_imgs, str(i) + '.jpg')).is_file():
                im = cv2.imread(os.path.join(path_imgs, str(i) + '.jpg'))
            else:
                im = cv2.imread(os.path.join(path_imgs, str(i) + '.png'))
            # print(im.shape)

            # exit()

            kpt, id_lists = load_poses_from_json1(os.path.join(path_jsons, str(i) + '.json'))

            im_kpts = im.copy()

            im_pose = im.copy()

            yaw_list, pitch_list, roll_list, yaw_u_list, pitch_u_list, roll_u_list = [], [], [], [], [], []
            center_xy = []

            for j, kpt_person in enumerate(kpt):

                im_kpts = draw_key_points_pose(im_kpts, kpt_person, True)
                # exit()
                # print(kpt_person)
                #TODO here change order if openpose
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

            t = os.path.join(result_folder, fold)
            path_json = os.path.join(t, 'ypr', str(i) + '.json')

            ypr_to_json(path_json, yaw_list, pitch_list, roll_list, yaw_u_list, pitch_u_list, roll_u_list, id_lists, center_xy)
            # (path_json, norm_list, unc_list, ids_list, suffix_)

            # draw_cones(yaw.numpy(), pitch.numpy(), roll.numpy(), unc_yaw.numpy(), unc_pitch.numpy(), unc_roll.numpy(), im_pose, tdx, tdy)
