% Eli Bowen 5/2022
% executes the supplied julia call, and returns any response
% requires that julia is all set up on your computer for command line execution
% USAGE
%   x = Julia(fullfile(ComputerProfile.MatlabCommonDir(), '..', 'JuliaCommon', 'ComputerProfile.jl'), 'DatasetDir')
% INPUTS
%   jlFile: (char) .jl file containing a function named by the next argument (full path preferred)
%   func:   (char) function name
%   varargin
% RETURNS
%   whatever the julia function returns
function [varargout] = Julia(jlFile, func, varargin)
    validateattributes(jlFile, {'char'}, {'nonempty'}, 1);
    validateattributes(func, {'char'}, {'nonempty'}, 2);
    assert(endsWith(jlFile, '.jl'), 'jlFile must end with .jl');

    file = CacheFile(varargin, func, 'julia');
    inFile = strrep(file, '.mat', '_in.h5');
    outFile = strrep(file, '.mat', '_out.h5');
    if isfile(inFile)
        delete(inFile);
    end
    if isfile(outFile)
        delete(outFile);
    end
    for i = 1 : numel(varargin)
        if ~isempty(varargin{i})
            h5create(inFile, ['/group/ds',num2str(i)], size(varargin{i}), 'Datatype', class(varargin{i})); % no compression by default
            h5write(inFile, ['/group/ds',num2str(i)], varargin{i});
        end
    end

    command = ['julia --threads=',num2str(DetermineNumJavaComputeCores()),' ',fullfile(ComputerProfile.MatlabCommonDir(), 'JuliaFromMatlab.jl'),' "',jlFile,'" "',func,'" "',inFile,'" "',outFile,'"'];
    disp(command);
    [status,cmdout] = system(command);
    if status ~= 0
        error(cmdout);
    end

    if isfile(inFile)
        delete(inFile);
    end

    varargout = {};
    if isfile(outFile)
        info = h5info(outFile);
        varargout = cell(1, numel(info.Groups.Datasets));
        for i = 1 : numel(info.Groups.Datasets)
            varargout{i} = h5read(outFile, ['/group/ds',num2str(i)]);
        end
        delete(outFile);
    end
end