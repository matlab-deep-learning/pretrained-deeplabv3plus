classdef DownloadDeeplabV3PlusFixture < matlab.unittest.fixtures.Fixture
    % DownloadDeeplabFixture   A fixture for calling downloadPretrainedDeepLabV3Plus if
    % necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.
    
    % Copyright 2021 The MathWorks, Inc
    
    properties(Constant)
        DeeplabV3DataDir = fullfile(getRepoRoot(),'model')
    end
    
    properties
        DeeplabV3Exist (1,1) logical        
    end
    
    methods
        function setup(this)            
            this.DeeplabV3Exist = exist(fullfile(this.DeeplabV3DataDir,'deepLabV3Plus-voc'),'file')==2;
            
            % Call this in eval to capture and drop any standard output
            % that we don't want polluting the test logs.
            if ~this.DeeplabV3Exist
            	evalc('helper.downloadPretrainedDeepLabv3Plus();');
            end       
        end
        
        function teardown(this)
            if this.DeeplabV3Exist
            	%delete(fullfile(this.Yolov2DataDir,'model','darknet19-voc'));
            end            
        end
    end
end