%% Semantic Segmentation Using DeepLabv3+ Network
% The following code demonstrates running semantic segmentation on a pre-trained 
% DeepLabv3+ network, trained on PASCAL VOC dataset.

%% Prerequisites
% To run this example you need the following prerequisites - 
% # MATLAB (R2020a or later) with Computer Vision and Deep Learning Toolbox.
% # Pretrained DeepLabv3+ network (download instructions below).

%% Download the Pre-trained Network
model = helper.downloadPretrainedDeepLabv3Plus;
net = model.net;

%% Perform Semantic Segmentation Using DeepLabv3+ Network
% Read test image.
image = imread('visionteam.jpg');

% Resize the image such that its smaller dimension is scaled to 513.
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


% Copyright 2021 The MathWorks, Inc.