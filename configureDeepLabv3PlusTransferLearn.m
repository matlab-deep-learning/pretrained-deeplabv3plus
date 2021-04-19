%% Configure Pretrained DeepLabv3+ Network for Transfer Learning
% The following code demonstrates configuring a pretrained 
% DeepLabv3+[1] network on the custom dataset.

%% Download Pretrained Model
model = helper.downloadPretrainedDeepLabv3Plus;
net = model.net;

%% Download CamVid Dataset
% This example uses the CamVid dataset[2] from the University of Cambridge for training. 
% This dataset is a collection of images containing street-level views obtained while 
% driving. The dataset provides pixel-level labels for 32 semantic classes including car, 
% pedestrian, and road.
%
% Download the CamVid dataset from the following URLs.
imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';
 
outputFolder = fullfile(tempdir,'CamVid'); 
labelsZip = fullfile(outputFolder,'labels.zip');
imagesZip = fullfile(outputFolder,'images.zip');

if ~exist(labelsZip, 'file') || ~exist(imagesZip,'file')   
    mkdir(outputFolder)
       
    disp('Downloading 16 MB CamVid dataset labels...'); 
    websave(labelsZip, labelURL);
    unzip(labelsZip, fullfile(outputFolder,'labels'));
    
    disp('Downloading 557 MB CamVid dataset images...');  
    websave(imagesZip, imageURL);       
    unzip(imagesZip, fullfile(outputFolder,'images'));    
end

% Note: Download time of the data depends on your Internet connection. The commands 
% used above block MATLAB until the download is complete. Alternatively, you can 
% use your web browser to first download the dataset to your local disk. To use 
% the file you downloaded from the web, change the 'outputFolder' variable above 
% to the location of the downloaded file.

%% Load CamVid Images
imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);

%% Load CamVid Pixel-Labeled Images
% To make training easier, the 32 original classes in CamVid are grouped into 
% 11 classes as follows. To reduce 32 classes into 11, multiple classes from the 
% original dataset are grouped together. For example, "Car" is a combination of 
% "Car", "SUVPickupTruck", "Truck_Bus", "Train", and "OtherMoving".
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

% Return the grouped label IDs by using the helper function 'camvidPixelLabelIDs'.
labelIDs = helper.camvidPixelLabelIDs;

% Use the classes and label IDs to create the pixelLabelDatastore.
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

%% Analyze Dataset Statistics
% To see the distribution of class labels in the CamVid dataset, use 'countEachLabel'. 
% This function counts the number of pixels by class label.
tbl = countEachLabel(pxds);

% Ideally, all classes would have an equal number of observations. However, 
% the classes in CamVid are imbalanced, which is a common issue in automotive 
% data-sets of street scenes. Such scenes have more sky, building, and road pixels 
% than pedestrian and bicyclist pixels because sky, buildings and roads cover 
% more area in the image. If not handled correctly, this imbalance can be detrimental 
% to the learning process because the learning is biased in favor of the dominant 
% classes. To handle this issue, class weighting has been used.

%% Prepare Training, Validation, and Test Sets
% Deeplabv3+ is trained using 60% of the images from the dataset. The rest 
% of the images are split evenly in 20% and 20% for validation and testing 
% respectively. The following code randomly splits the image and pixel label 
% data into a training, validation and test set.
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = helper.partitionCamVidData(imds,pxds);

%% Configure Pretrained Network
% To configure the DeepLabv3+ network for transfer learning, you should replace 
% the last convolutional layer and pixelClassificationLayer in the layergraph 
% obtained from the pretrained model.

% Specify the number of classes.
numClasses = numel(classes);

% Extract the layergraph from the pretrained network to perform custom
% modification.
lgraph = layerGraph(net);

% Replace the last convolution layer in the pretrained network with the new 
% convolution layer.
convLayer = convolution2dLayer([1 1], numClasses,'Name', 'node_398');
lgraph = replaceLayer(lgraph,"node_398",convLayer);

% Balance classes using class weighting.
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

% Replace the pixel classification layer in the pretrained network with the classweights
% and new pixel classification layer.
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"labels",pxLayer);

% Use analyzeNetwork to visualize the new network.
analyzeNetwork(lgraph);

%% Data Augmentation
% Data augmentation is used to improve network accuracy by randomly transforming 
% the original data during training. By using data augmentation, you can add 
% more variety to the training data without increasing the number of labeled 
% training samples. 
%
% This pretrained model has input size of [513,513,3] and the CamVid images
% are of size [720,960,3]. Hence, it would be better to use random patches 
% of size [513,513,3] from the given input images for training.
%
% In this case, 'randomPatchExtractionDatastore' is useful for creating 
% such training and validation datastores. 
% 
% To apply the same random transformation to both image and pixel label data 
% use 'imageDataAugmenter' object in 'DataAugmentation' NVP during creating 
% 'randomPatchExtractionDatastore' object. Here, random left/right reflection 
% and random X/Y translation of +/- 10 pixels is used for data augmentation.
xTrans = [-10 10];
yTrans = [-10 10];

augmenter = imageDataAugmenter('RandXReflection',true, 'RandXTranslation',xTrans, 'RandYTranslation',yTrans);
dsTrain = randomPatchExtractionDatastore(imdsTrain,pxdsTrain,[513 513],'PatchesPerImage',8, 'DataAugmentation', augmenter);

% Note that data augmentation is not applied to the test and validation data. 
% Ideally, test and validation data should be representative of the original 
% data and is left unmodified for unbiased evaluation.

%% Select Training Options
% The optimization algorithm used for training is stochastic gradient descent 
% with momentum (SGDM). Use trainingOptions to specify the hyper-parameters 
% used for SGDM.

% Define validation datastore.
dsVal = randomPatchExtractionDatastore(imdsVal,pxdsVal,[513 513],'PatchesPerImage',8);

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',6, ...  
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);

% The learning rate uses a piecewise schedule. The learning rate is reduced 
% by a factor of 0.3 every 10 epochs. This allows the network to learn quickly 
% with a higher initial learning rate, while being able to find a solution 
% close to the local optimum once the learning rate drops.
%
% The network is tested against the validation data every epoch by setting 
% the 'ValidationData' parameter. The 'ValidationPatience' is set to 4 to 
% stop training early when the validation accuracy converges. This prevents 
% the network from overfitting on the training dataset.
%
% A mini-batch size of 16 is used for training. You can increase or decrease 
% this value based on the amount of GPU memory you have on your system.
%
% In addition, 'CheckpointPath' is set to a temporary location. This name-value 
% pair enables the saving of network checkpoints at the end of every training 
% epoch. If training is interrupted due to a system failure or power outage, 
% you can resume training from the saved checkpoint. Make sure that the location 
% specified by 'CheckpointPath' has enough space to store the network checkpoints.


% Now, you can pass the 'dsTrain', 'lgraph' and 'options' to trainNetwork 
% as shown in 'Start Training' section of the example 'Semantic Segmentation 
% Using Deep Learning' to obtain deepLabv3+ model trained on the custom dataset.
%
% You can follow the sections 'Test Network on One Image' for inference using 
% the trained model and 'Evaluate Trained Network' for evaluating metrics.


%% References

% [1] Chen, Liang-Chieh et al. “Encoder-Decoder with Atrous Separable Convolution 
% for Semantic Image Segmentation.” ECCV (2018).
% 
% [2] Brostow, G. J., J. Fauqueur, and R. Cipolla. "Semantic object classes 
% in video: A high-definition ground truth database." Pattern Recognition Letters. 
% Vol. 30, Issue 2, 2009, pp 88-97.
% 
% Copyright 2021 The MathWorks, Inc.