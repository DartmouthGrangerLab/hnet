% Eli Bowen 5/1/2021
% code draws a line using the Bresenham line algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
% very fast, no anti-aliasing
% based on 2 functions:
% 1) by CHANDAN KUMAR:  https://www.mathworks.com/matlabcentral/fileexchange/25544-line-drawing-by-bresenham-algorithm
% 2) by Aaron Wetzler:  https://www.mathworks.com/matlabcentral/fileexchange/28190-bresenham-optimized-for-matlab
% pix = DrawSkinnyLineSqr([1,1], [8,3]);
% figure; [c,r] = meshgrid(0:10, 0:10); scatter(c(:), r(:), 'r.'); hold on; scatter(pix(:,2), pix(:,1), 50, 'k*'); axis([0,10,0,10]); xlabel('row'); ylabel('col');
% INPUTS
%   pt1         - 1 x 2 (numeric) 2D position of point 1 (x,y or r,c - whatever - pix coordinates returned with same dimensionality)
%   pt2         - 1 x 2 (numeric) 2D position of point 2
%   scaleFactor - scalar (numeric)
% RETURNS
%   pix - n_pixels x 2 (numeric) 2D coordinates of pixels that should be illuminated by the line
% see also DrawSkinnyLineHex
function pix = DrawSkinnyLineSqr(pt1, pt2, scaleFactor)
    validateattributes(pt1,         {'numeric'}, {'nonempty'}, 1);
    validateattributes(pt2,         {'numeric'}, {'nonempty'}, 2);
    validateattributes(scaleFactor, {'numeric'}, {'nonempty','scalar'}, 3);

    %% first, set start and end points to be real pixels
    pt1 = floor(pt1 ./ scaleFactor);
    pt2 = floor(pt2 ./ scaleFactor);

    is_steep = (abs(pt2(2) - pt1(2)) > abs(pt2(1) - pt1(1)));

    %% reorient line if necessary
    if is_steep % convert to non-steep by flipping coordinates
        pt1 = flip(pt1); % x is y, y is x
        pt2 = flip(pt2);
    end
    if pt1(1) > pt2(1) % make sure first point has smallest x position (for consistency, and because ensures sign of dx is always +)
        temp = pt1;
        pt1 = pt2;
        pt2 = temp;
    end

    %% prep
    s = sign(pt2 - pt1); % s(1) always +1, s(2) ensures correct line slope
    d = abs(pt2 - pt1);

    %% draw line
    pix = zeros(numel(pt1(1):pt2(1)), 2, 'like', pt1);
    pix(:,1) = pt1(1):pt2(1);
    if d(2) == 0
        pix(:,2) = pt1(2);
    else
        % v1
        y = pt1(2); % init y
        param = 2*d(2) - d(1); % init error parameter
        for i = 1 : size(pix, 1) % loop to travel along X
            pix(i,2) = y;
            param = param + 2*d(2);
            if param > 0 % if parameter value is exceeded
                y = y + s(2); % increase y coordinate
                param = param - 2*d(1); % decrease parameter value
            end
        end
        % v2 (faster; untested)
%         q = [0;diff(mod((floor(d(1)/2):-d(2):-d(2)*d(1)+floor(d(1)/2))', d(1))) >= 0];
%         pix(:,2) = pt1(2) + s(2) .* cumsum(q);
    end

    % re-orient
    if is_steep
        temp = pix(:,1); % swap x and y
        pix(:,1) = pix(:,2);
        pix(:,2) = temp;
    end

    pix = pix .* scaleFactor;
end