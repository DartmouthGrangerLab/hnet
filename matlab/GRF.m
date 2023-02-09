% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% graph type enum
classdef GRF < uint8
    enumeration
        NULL   (0) % n/a, null
        GRID1D (1) % 1d grid
        GRID2D (2) % 2d rectangular grid
        GRID2DMULTICHAN (3) % 2d rectangular grid w >1 channel
        FULL   (4) % fully connected
        SELF   (5) % each node is connected only to itself
    end
end