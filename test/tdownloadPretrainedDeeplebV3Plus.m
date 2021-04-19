classdef(SharedTestFixtures = {DownloadDeeplabV3PlusFixture}) tdownloadPretrainedDeeplebV3Plus < matlab.unittest.TestCase
    % Test for downloadPretrainedDeeplebV3Plus
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture DownloadDeeplabV3PlusFixture calls
    % downloadPretrainedDeeplebV3Plus. Here we check that the downloaded files
    % exists in the appropriate location.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'model');
    end
    
    methods(Test)
        function verifyDownloadedFilesExist(test)
            dataFileName = 'deepLabV3Plus-voc.mat';
            test.verifyTrue(isequal(exist(fullfile(test.DataDir,dataFileName),'file'),2));
        end
    end
end
