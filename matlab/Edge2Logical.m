% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   x - n_edges x n (EDG enum) edges, as returned by GetRelations
%   do_include_na - OPTIONAL scalar (logical) (default = true)
% RETURNS
%   y - n_edges*EDG.n x n (logical)
function y = Edge2Logical(x, do_include_na)
    if ~exist('do_include_na', 'var') || isempty(do_include_na)
        do_include_na = true;
    end
    [n_edges,n] = size(x);

    y = false(n_edges, n, EDG.n);
    uniqEdgeTypes = enumeration('EDG');
    
    for i = 1 : EDG.n
        y(:,:,i) = (x == uniqEdgeTypes(i));
    end
    
    if ~do_include_na
        y(:,:,1) = []; % remove n/a
    end
    
    y = permute(y, [1,3,2]);
    y = reshape(y, n_edges * size(y, 2), n);
end