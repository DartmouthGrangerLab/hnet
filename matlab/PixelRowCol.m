% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% essentially this is meshgrid()
% INPUTS
%   imgSz - 3 x 1 (int-valued numeric)
% RETURNS
%   row - 1 x n_pixels (numeric)
%   col - 1 x n_pixels (numeric)
function [row,col] = PixelRowCol(imgSz)
    arguments
        imgSz (3,1)
    end
    
    row = mod(0:imgSz(1)*imgSz(2)-1, imgSz(1)); % convert to col
    col = floor((0:imgSz(1)*imgSz(2)-1) ./ imgSz(1)); % convert to row

    row = row + 1; % was zero-based indexing
    col = col + 1; % was zero-based indexing
    
    % above is verified identical to below
%     [col,row] = meshgrid(1:imgSz(2), 1:imgSz(1));
%     row = row(:)';
%     col = col(:)';
end