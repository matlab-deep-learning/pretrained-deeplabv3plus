function model = downloadPretrainedDeepLabv3Plus()
% The downloadPretrainedDeepLabv3Plus function loads a pretrained
% DeepLabv3Plus network.
%
% Copyright 2021 The MathWorks, Inc.

dataPath = 'model';
modelName = 'deepLabV3Plus-voc';
netFileFullPath = fullfile(dataPath, modelName);

% Add '.mat' extension to the data.
netFileFull = [netFileFullPath,'.zip'];

if ~exist(netFileFull,'file')
    fprintf(['Downloading pretrained', modelName ,'network.\n']);
    fprintf('This can take several minutes to download...\n');
    url = 'https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/deepLabV3Plus/deepLabV3Plus-voc.zip';
    websave (netFileFullPath,url);
    unzip(netFileFullPath, dataPath);
    model = load([dataPath, '/deepLabV3Plus-voc.mat']);
else
    fprintf('Pretrained DeepLabv3Plus network already exists.\n\n');
    unzip(netFileFullPath, dataPath);
    model = load([dataPath, '/deepLabV3Plus-voc.mat']);
end
end