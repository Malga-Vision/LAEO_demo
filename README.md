# LAEO DEMO
### Author: [Federico Figari Tomenotti](https://github.com/Fede1995) <br />
<br />

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

The models can't be uploaded on github
