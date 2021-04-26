%% Code generation For DeepLabv3+ Network
% The following script demonstrates how to perform code generation for a pretrained 
% DeepLabv3+ semantic segmentation network, trained on PASCAL VOC dataset.

%% Download the Pre-trained Network
helper.downloadPretrainedDeepLabv3Plus;

%% Preprocess the input image
% Read test image.
image = imread('visionteam.jpg');

% Resize the image such that its smaller dimension is scaled to 513.
sz = size(image);
[~,k] = min(sz(1:2));
scale = 513/sz(k);
img  = imresize(image, scale, "bilinear");
newSz = size(img);

%% Run MEX code generation
% The deepLabv3Plus_predict.m is entry-point function that takes an input image
% and gives output. The function uses a persistent object deepLabv3PlusObj to 
% load the DAG network object and reuses the persistent object for prediction 
% on subsequent calls.
%
% To generate CUDA code for the deepLabv3Plus_predict entry-point function, 
% create a GPU code configuration object for a MEX target and set the 
% target language to C++. 
% 
% Use the coder.DeepLearningConfig (GPU Coder) function to create a CuDNN 
% deep learning configuration object and assign it to the DeepLearningConfig 
% property of the GPU code configuration object. 
% 
% Run the codegen command and specify the input size. 
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
codegen -config cfg deepLabv3Plus_predict -args {ones(newSz(1),newSz(2),3,'uint8')} -report


%% Perform Semantic Segmentation Using Generated MEX 
% Call deepLabv3Plus_predict_mex on the input image.
predict_scores = deepLabv3Plus_predict_mex(img);

% The predict_scores variable is a three-dimensional matrix that has 21 channels 
% corresponding to the pixel-wise prediction scores for every class. 
% Compute the channel by using the maximum prediction score to get pixel-wise labels.
[~,argmax] = max(predict_scores,[],3);

% Overlay the segmented labels.
overlay = labeloverlay(img , argmax, 'Transparency', 0.4);

% Visualize the input and the result.
overlay = imresize(overlay, sz(1:2), 'bilinear');
montage({image, overlay});


% Copyright 2021 The MathWorks, Inc.