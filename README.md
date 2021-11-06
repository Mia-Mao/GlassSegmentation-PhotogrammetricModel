# GlassSegmentation-PhotogrammetricModel
This repository contains the code for the paper "Glass Segmentation from 3D Building with Facade Information and Multi-view Oblique Images"

### Abstract
Glass segmentation from 3D urban scene provides guidance for wide applications, such as sunlight analysis, light pollution assessment, energy conservation, etc. However, most advanced algorithms and datasets are proposed to detect glass in natural images for indoor navigation, and few of them focus on glass in the 3D city scene. In this paper, we present a two-stage pipeline to detect glass from building in 3D urban scene using façades information and multi-view oblique images. In stage I, we apply an ensemble module to detect glass from oblique images, considering the spatial structural relationship and edge feature of glass pieces. In stage II, we build a correspondence between multi-view images and 3D triangle mesh to distinguish the building façades (glass or non-glass (3D)). The façades are extracted via point density and height gradient maps basing on the truth that a typical structure of them usually vertically aligned to the ground surface. A dataset is constructed for detector learning and method validation. Experiments show the ensemble method performs superiority and demonstrate our workflow is feasible to segment glass in the 3D scene. \textbf{The dataset can be accessed at}

### Citation
If you use this code or our dataset (including test set), please cite:

```


```
### Dataset
(1) Request dataset via Email 
please request access to the dataset for non-commercial use via email (maoz@whu.edu.cn). Please use your official university/company email address. Thank you!

(2) Access to the dataset via BaiduNetdisk
Link: https://pan.baidu.com/s/1RfuFxwxH6bAWv3_rmmKwVw  Extract code: zpdy

### Test
Download the code and run main.py


### Evaluation
Run Evaluation.py
