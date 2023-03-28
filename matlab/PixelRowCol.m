% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% essentially this is meshgrid()
% INPUTS
%   imgsz - 3 x 1 (int-valued numeric) image size (ignores num channels)
% RETURNS
%   row - n_pixels x 1 (numeric)
%   col - n_pixels x 1 (numeric)
function [row,col] = PixelRowCol(imgsz)
    arguments
        imgsz(3,1)
    end
    
    row = mod(0:imgsz(1)*imgsz(2)-1, imgsz(1))'; % convert to col
    col = floor((0:imgsz(1)*imgsz(2)-1) ./ imgsz(1))'; % convert to row

    row = row + 1; % was zero-based indexing
    col = col + 1; % was zero-based indexing
    
    % above is verified identical to below
%     [col,row,chan] = meshgrid(1:imgsz(2), 1:imgsz(1));
%     row = row(:);
%     col = col(:);

    row = repmat(row, [imgsz(3),1]);
    col = repmat(col, [imgsz(3),1]);
end