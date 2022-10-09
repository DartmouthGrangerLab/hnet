% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   m - 2D (numeric)
%   d - scalar (int-valued numeric)
% RETURNS
%   pmatrix
function pmatrix = PermutationMat(M, d)
    arguments
        M(:,:) {mustBeNumeric}, d(1,1) {mustBeNumeric}
    end

    [x,y] = meshgrid(-floor(d/2):floor(d/2)+mod(d, 2)-1,-floor(d/2):floor(d/2)+mod(d, 2)-1);
    z = ones(size(x));
    coords = cat(2, x(:), y(:), z(:)); % ? x 3
    newCoords = round(M * coords')';

    coords = coords(:,1:2);
    newCoords = newCoords(:,1:2);

    minX = min(coords(:));
    
    % only keep valid indices
    mask = all((newCoords >= minX) & (newCoords <= max(coords)), 2);
    coords = coords(mask,:);
    newCoords = newCoords(mask,:);
    
    coords = coords - minX;
    newCoords = newCoords - minX;
    
    % convert from coordinates to linear indices
    from_coord = coords(:,1).*d + coords(:,2) + 1;
    to_coord   = newCoords(:,1).*d + newCoords(:,2) + 1;
    
    pmatrix = zeros(d*d, d*d);
    for i = 1 : numel(from_coord)
        pmatrix(from_coord(i),to_coord(i)) = 1;
    end
end