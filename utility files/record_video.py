import argparse
import os
import numpy as np
from tqdm import tqdm
from src.camera.intialize_camera import initialize_zed_camera
from src.utils.my_utils import initialize_video_recorder
import logging
import pyzed.sl as sl


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger_format = logging.Formatter("%(asctime)s::[%(levelname)s]::%(name)s::%(filename)s::%(lineno)d: %(message)s")
handler.setFormatter(logger_format)
logger.addHandler(handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=str, default=None, help="path to video output", required=True)
    ap.add_argument("-od", "--output-depth", type=str, default=None, help="path to video depth output", required=False)
    ap.add_argument("-odm", "--output-point-cloud", type=str, default=None, help="path to numpy array containing point cloud information", required=False)
    config = ap.parse_args()

    zed, runtime_parameters = initialize_zed_camera()
    image, depth_image, point_cloud = sl.Mat(), sl.Mat(), sl.Mat()
    fps, shape = 10, (1280, 720)
    writer, writer_depth = initialize_video_recorder(os.path.abspath(config.output), os.path.abspath(config.output_depth), fps, shape)
    point_cloud_list = []

    num_frames = 200

    logger.info("Start recording")
    for _ in tqdm(range(num_frames)):
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  # A new image is available if grab() returns SUCCESS

            zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve left image
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)  # Retrieve depth map measure. Depth is aligned on the left image

            img = image.get_data()[:, :, :3]
            depth_img = depth_image.get_data()[:, :, :3]
            point_cloud_img = point_cloud.get_data()[:, :, :3]

            writer.write(img)
            writer_depth.write(depth_img)
            point_cloud_list.append(point_cloud_img)

    logger.info("End recording")
    zed.close()

    logger.info("Save point cloud")

    if os.path.isdir(os.path.split(os.path.abspath(config.output_point_cloud))[0]):
        np.save(config.output_point_cloud, point_cloud_list, allow_pickle=True)
    else:
        logger.error("Invalid path for the point cloud list; folder does not exist")
        exit(1)

    logger.info("End saving point cloud")
    exit(0)





