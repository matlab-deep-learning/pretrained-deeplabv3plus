classdef(SharedTestFixtures = {DownloadDeeplabV3PlusFixture}) tload < matlab.unittest.TestCase
    % Test for loading the downloaded models.
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture DownloadDeeplabV3PlusFixture calls
    % downloadPretrainedDeeplabV3Plus. Here we check that the properties of
    % downloaded models.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'model');        
    end
    
    methods(Test)
        function verifyModelAndFields(test)
            % Test point to verify the fields of the downloaded models are
            % as expected.
                                    
            loadedModel = load(fullfile(test.DataDir,'deepLabV3Plus-voc.mat'));
            
            test.verifyClass(loadedModel.net,'DAGNetwork');
            test.verifyEqual(numel(loadedModel.net.Layers),376);
            test.verifyEqual(size(loadedModel.net.Connections),[416 2])
            test.verifyEqual(loadedModel.net.InputNames,{'Input'});
            test.verifyEqual(loadedModel.net.OutputNames,{'labels'});            
        end        
    end
end