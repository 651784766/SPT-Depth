# Dense Monocular Depth Estimation for Stereoscopic Vision Based on Pyramid Transformer and Multi-Scale Feature Fusion

![pytorch](https://img.shields.io/badge/pytorch-v1.10-green.svg?style=plastic)
![wandb](https://img.shields.io/badge/wandb-v0.12.10-blue.svg?style=plastic)
![scipy](https://img.shields.io/badge/scipy-v1.7.3-orange.svg?style=plastic)

<!-- ![presentation](https://i.ibb.co/rbySmMc/DL-FOD-POSTER-1.png) -->

<p align="center">
  <img src="images/pull_figure.png"/>
</p>

<!-- > Input image taken from: https://koboguide.com/how-to-improve-portrait-photography/ -->

## Abstract

<!-- Recent works have shown that in the real world, humans
rely on the image obtained by their left and right eyes in order to estimate depths of surrounding objects. Thus, -->
>Stereoscopic display technology plays a significant role in industries, such as film, television and autonomous driving. The accuracy of depth estimation is crucial for achieving high-quality and realistic stereoscopic display effects. In addressing the inherent challenges of applying Transformers to depth estimation, the Stereoscopic Pyramid Transformer-Depth (SPT-Depth)


## :pushpin: Requirements

Run: ``` pip install -r requirements.txt ```

## :rocket: Running the model

You can first download one of the models from the model:

### :bank: Model zoo

Get the links of the following models:

+ [```SPT-large.p```]
+ Other models coming soon...

And put the ```.p``` file into the directory ```models/```. After that, you need to update the ```config.json``` ([Tutorial here](https://github.com/antocad/FocusOnDepth/wiki/Config-Wiki)) according to the pre-trained model you have chosen to run the predictions (this means that if you load a depth-only model, then you have to set ```type``` to ```depth``` for example ...).

### :dart: Run a prediction

Put your input images (that have to be ```.png``` or ```.jpg```) into the ```input/``` folder. Then, just run ```python run.py``` and you should get the depth maps as well as the segmentation masks in the ```output/``` folder.


## :hammer: Training

### :wrench: Build the dataset

Our model is trained on a combination of
+ [inria movie 3d dataset](https://www.di.ens.fr/willow/research/stereoseg/) | [view on Kaggle](https://www.kaggle.com/antocad/inria-fod/)
+ [NYU2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | [view on Kaggle](https://www.kaggle.com/antocad/nyuv2-fod)
+ [PoseTrack](https://posetrack.net/) | [view on Kaggle](https://www.kaggle.com/antocad/posetrack-fod)

### :pencil: Configure ```config.json```

Specific configurations are given in the paper

### :nut_and_bolt: Run the training script
After that, you can simply run the training script: ```python train.py```


## :scroll: Citations

Our work is based on Ranflt et al. Unlike them we will focus on autostereoscopic vision. This research is based on an 8K naked eye 3D project, and all subsequent code will be given after the corresponding patents are filed, thank you for your support!