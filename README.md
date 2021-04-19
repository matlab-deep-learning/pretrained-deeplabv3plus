# Pretrained DeepLabv3+ Network for Semantic Segmentation

This repository provides a pretrained DeepLabv3+[1] semantic segmentation model for MATLAB&reg;.

Requirements
------------

- MATLAB&reg; R2020a or later.
- Deep Learning Toolbox&trade;.
- Computer Vision Toolbox&trade;.

Overview
--------

Semantic segmentation is a computer vision technique for segmenting different classes of objects in images or videos. This pretrained network is trained using PASCAL VOC dataset[2] which have 20 different classes including airplane, bus, car, train, person, horse etc. 

For more information about semantic segmentation, see [Getting Started with Semantic Segmentation Using Deep Learning](https://mathworks.com/help/vision/ug/getting-started-with-semantic-segmentation-using-deep-learning.html).

 
Getting Started
---------------
Download or clone this repository to your machine and open it in MATLAB&reg;.

### Download the pretrained network
Use the below helper to download the pretrained network.

```
model = helper.downloadPretrainedDeepLabv3Plus;
net = model.net;
```

Semantic Segmentation Using Pretrained DeepLabv3+
-------------------------------------------------

```
% Read test image from images folder
image = imread('visionteam.jpg');

% Resize the image to the size used to train the network. 
% The image is resized such that smallest dimension is 513.
sz = size(image);
[~,k] = min(sz(1:2));
scale = 513/sz(k);
img  = imresize(image, scale, "bilinear");

% Use semanticseg function to generate segmentation map.
result = semanticseg(img, net);

% Generate the overlaid result using generated map.
overlay = labeloverlay(img , result, 'Transparency', 0.4);

% Visualize the input and the result.
overlay = imresize(overlay, sz(1:2), 'bilinear');
montage({image, overlay});
```
Left-side image is the input and right-side image is the corresponding segmentation output.

![alt text](images/result.png?raw=true)


Train Custom DeepLabv3+ Using Transfer Learning
-----------------------------------------------
Transfer learning enables you to adapt a pretrained DeepLabv3+ network to your dataset. Create a custom DeepLabv3+ network for transfer learning with a new set of classes using the `configureDeepLabv3PlusTransferLearn.m` script. For more information about training a DeepLabv3+ network, see [Semantic Segmentation Using Deep Learning](https://www.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html)


Code Generation for DeepLabV3+
------------------------------
Code generation enables you to generate code and deploy DeepLabv3+ on multiple embedded platforms.

Run `codegenDeepLabv3Plus.m`. This script calls the `deepLabv3Plus_predict.m` entry point function and generate CUDA code for it. It will run the generated MEX and gives output.

| Model | Inference Speed (FPS) | 
| ------ | ------ | 
| DeepLabv3Plus w/o codegen | 3.5265 |
| DeepLabv3Plus with codegen | 21.5526 |

- Performance (in FPS) is measured on a TITAN-RTX GPU using 513x513 image.

For more information about codegen, see [Deep Learning with GPU Coder](https://www.mathworks.com/help/gpucoder/gpucoder-deep-learning.html)


Accuracy
--------
Metrics are mIoU, global accuracy and mean accuracy computed over 2012 PASCAL VOC val data. 

| Model | mIoU | Global Accuracy | Mean Accuracy | Size (MB) | Classes |
| ------ | ------ | ------ | ------ | ------ | ------ |
| DeepLabv3Plus-VOC | 0.77299 | 0.94146 | 0.87279 | 209 | [voc class names](+helper/pascal-voc-classes.txt) |

- During computation of these metrics, val images are first resized such that the smaller dimension of the images are scaled to 513 because that matches the training preprocessing and then a center crop of size 513x513 is used for evaluation.


References
-----------
[1] Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European conference on computer vision (ECCV). 2018.

[2] The PASCAL Visual Object Classes Challenge: A Retrospective Everingham, M., Eslami, S. M. A., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A. International Journal of Computer Vision, 111(1), 98-136, 2015.


Copyright 2021 The MathWorks, Inc.
