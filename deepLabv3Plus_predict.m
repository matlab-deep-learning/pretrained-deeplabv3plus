function out = deepLabv3Plus_predict(in)
%#codegen
% Copyright 2021 The MathWorks, Inc.

persistent deepLabv3PlusObj;

if isempty(deepLabv3PlusObj)
    deepLabv3PlusObj = coder.loadDeepLearningNetwork('model/deepLabV3Plus-voc.mat');
end

% Pass input.
netInputSize = [513,513,3];

if isequal(size(in), netInputSize)
    out = predict(deepLabv3PlusObj,in);
else
    out = activations(deepLabv3PlusObj,in,'labels');
end