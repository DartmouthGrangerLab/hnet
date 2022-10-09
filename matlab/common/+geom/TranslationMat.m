% 2D translation matrix
% INPUTS
%   cx - scalar (numeric)
%   cy - scalar (numeric)
% RETURNS
%   x - 3 x 3 (numeric)
function x = TranslationMat(cx, cy)
    x = [1,0,cx;...
         0,1,cy;...
         0,0,1];
end