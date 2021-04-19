function out = deepLabv3Plus_predict(in)
%#codegen
% Copyright 2021 The MathWorks, Inc.

persistent deepLabv3PlusObj;

if isempty(deepLabv3PlusObj)
    deepLabv3PlusObj = coder.loadDeepLearningNetwork('model/deepLabV3Plus-voc.mat');
end

% Pass input.
out = predict(deepLabv3PlusObj,in);