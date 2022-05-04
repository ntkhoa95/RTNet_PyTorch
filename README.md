# RTNet_PyTorch
Implementation of RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes (IEEE RAL) for RGB-D dataset using PyTorch
This is the unofficial implementation using PyTorch of [RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes](https://github.com/yuxiangsun/RTFNet/blob/master/doc/RAL2019_RTFNet.pdf) (IEEE RA-L) for evaluating RGB-D [Ground Mobile Robot Perception Dataset](https://github.com/hlwang1124/GMRPD)

## Introduction
RTFNet is a data-fusion network for semantic segmentation. It consists of two encoders and one decoder. RTFNet is well-designed for not only RGB-Thermal data but also RGB-D data. Please take a look at [paper](https://doi.org/10.1109/LRA.2019.2932874).

## To Use
```
python train.py --dataset gmrpd --experiment_name gmrpd_manual
```

## Citation
If you use RTFNet in an academic work, please cite:

```
@ARTICLE{sun2019rtfnet,
author={Yuxiang Sun and Weixun Zuo and Ming Liu}, 
journal={{IEEE Robotics and Automation Letters}}, 
title={{RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes}}, 
year={2019}, 
volume={4}, 
number={3}, 
pages={2576-2583}, 
doi={10.1109/LRA.2019.2904733}, 
ISSN={2377-3766}, 
month={July},}
```

If you use GMRPD Dataset, please cite:
```
@article{wang2021dynamic,
  title     = {Dynamic fusion module evolves drivable area and road anomaly detection: A benchmark and algorithms},
  author    = {Wang, Hengli and Fan, Rui and Sun, Yuxiang and Liu, Ming},
  journal   = {IEEE Transactions on Cybernetics},
  year      = {2021},
  publisher = {IEEE},
  doi       = {10.1109/TCYB.2021.3064089}
}
```

```
@article{wang2019self,
  title     = {Self-supervised drivable area and road anomaly segmentation using {RGB-D} data for robotic wheelchairs},
  author    = {Wang, Hengli and Sun, Yuxiang and Liu, Ming},
  journal   = {IEEE Robotics and Automation Letters},
  volume    = {4},
  number    = {4},
  pages     = {4386--4393},
  year      = {2019},
  publisher = {IEEE},
  doi       = {10.1109/LRA.2019.2932874}
}
```