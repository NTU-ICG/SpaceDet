# SpaceNet
SpaceNet: A Large-scale Realistic Space-based Image Dataset for Space Situational Awareness

> ❗ When using SpaceNet in your research, it is vital to cite both the SpaceNet dataset and the individual works that have contributed to your research. Accurate citation acknowledges the efforts and contributions of all researchers involved. For example, if your work involves a specific benchmark within SpaceNet, please include a citation for that benchmark apart from SpaceNet.

## Repository Overview

This repository provides the SpaceNet dataset, designed to advance research in Space Situational Awareness (SSA). SpaceNet offers realistic space-based images generated with accurate space orbit dynamics and a physical camera model, suitable for developing advanced SSA techniques. The dataset is split into three subsets: SpaceNet-100, SpaceNet-5000, and SpaceNet-full, catering to various image processing applications. This codebase is still under construction. Comments, issues, contributions, and collaborations are all welcomed!

## Data

SpaceNet provides a comprehensive and realistic space-based image dataset designed to advance research in Space Situational Awareness (SSA). The dataset includes high-resolution images of resident space objects (RSOs) captured from various orbits. Below are the available datasets and their respective download links:

- **SpaceNet-100** [Google Drive](#): A subset containing 100 high-resolution images for quick experimentation and algorithm testing (dataset for only one camera).
- **SpaceNet-5000** [Google Drive](#): A larger subset with 5000 images providing a more extensive dataset for detailed analysis and model training (dataset for only one camera).
- **SpaceNet-full** [Google Drive](#): The complete dataset including 781.5GB of images and 25.9MB of ground truth labels, designed for in-depth research and development of advanced SSA techniques (dataset for four cameras).

Our codebase accesses the datasets from ./data/ and benchmark codes from ./codes/ by default.

```plaintext
├── ...
├── data
│   ├── SpaceNet-100
│   ├── SpaceNet-5000
│   └── SpaceNet-full
├── codes
│   └── scripts
│   └── ultralytics
├── ...
```

## Training and evaluation scripts

We provide training and evaluation scripts for all the methods we support in [scripts folder](./codes/scripts).

## Supported Benchmarks

We compare the detection results of three object detection models: YOLOv5, YOLOv8, and YOLOv10. Each model is evaluated with five parameter sizes: n, s, m, l, and x.

- **YOLOv5** [GitHub](https://github.com/ultralytics/yolov5): YOLOv5, developed by Ultralytics, is a highly popular and efficient object detection model known for its speed and accuracy. It supports various tasks including object detection, instance segmentation, and more.

- **YOLOv8** [GitHub](https://github.com/ultralytics/ultralytics): YOLOv8 is an advanced version of YOLOv5, offering improved performance and new features. It continues to build on the strengths of its predecessor with enhanced detection capabilities and flexibility for different applications.

- **YOLOv10** [GitHub](https://github.com/THU-MIG/yolov10): YOLOv10, the latest in the YOLO series, introduces significant architectural improvements and optimizations. It is designed to deliver superior detection results with higher efficiency and accuracy compared to earlier versions.



