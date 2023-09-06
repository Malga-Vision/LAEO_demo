# LAEO DEMOs
### Author: [Federico Figari Tomenotti](https://github.com/Fede1995) <br />
<br />

## 1) Ready to try Demo 
It is hosted on **Hugging Face** at: [Head_Pose_Estimation_and_LAEO_computation](https://huggingface.co/spaces/FedeFT/Head_Pose_Estimation_and_LAEO_computation)
where you can use your webcam or images from your computer and try the Head Pose Estimation and the LAEO algorithm.
<br />

## 2) LAEO Zedcam Demo
It is more difficult to use, but support a ZedCam hardware as camera and also as Human Pose Estimator. 
Future project may involve taking into account the depth in the algorithm.

HW Requirements: <br />
- ZEDCAM <br />
- GPU (CPU it is fine for some features) <br />
<br />
Software:
Look at the requirements file <br />
Follow [ZED SDK] (https://www.stereolabs.com/docs/get-started-with-zed/#download-and-install-the-zed-sdkrl) and [Python ZED](https://www.stereolabs.com/docs/app-development/python/run/) <br />

<br /><br />
This program can be run using:
```bash
python3 demo_start.py -m zed [-f /folder/containing/file.svo]
```
or 
```bash
python3 demo_start.py -m centernet [-f /folder/containing/file.svo]
```

**m**: identifies the keypoints extractor algorithm\
**f**: a pre-recorded zedcam file, .svo format

The models can't be uploaded on github but you can find infos here where to download the Pose Estimator: [HHP-Net](https://github.com/Malga-Vision/HHP-Net)
