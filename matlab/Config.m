% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% constants and configuration params
classdef Config
    properties (Constant)
        OUT_DIR = fullfile('..', '..', 'output_matlab') % path to output directory (relative to matlab working dir)
        DATASET_DIR = fullfile('..', 'datasets') % path to dataset directory (relative to matlab working dir)
        MIN_EDGES_PER_CMP = 4 % minimum number of edges per component
        DO_CACHE = false
        DO_INVERT_COLORS = true % invert colors of nodes (true produces black text on white for mnist)
        DO_H_MODE = true % if true, generate and use Hamiltonians in Energy.m; if false, convert to a boolean feed-forward network in Energy.m
    end


    methods (Static)
        % returns the directory containing Config
        function x = MyDir()
            [x,~,~] = fileparts(mfilename("fullpath"));
        end
    end
end