% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   x - n_edges x n (EDG enum) edges, as returned by GetRelations
%   do_include_null - OPTIONAL scalar (logical) (default = true)
%   do_include_all_16 - OPTIONAL scalar (logical) (default = true)
% RETURNS
%   y - n_edges*EDG.n x n (logical)
function y = Edge2Logical(x, do_include_null, do_include_all_16)
    if ~exist("do_include_null", "var") || isempty(do_include_null)
        do_include_null = true;
    end
    if ~exist("do_include_all_16", "var") || isempty(do_include_all_16)
        do_include_all_16 = true;
    end
    [n_edges,n] = size(x);

    y = false(n_edges, n, EDG.n);
    uniqEdgeTypes = enumeration("EDG");
    
    for i = 1 : EDG.n
        y(:,:,i) = (x == uniqEdgeTypes(i));
    end

    if ~do_include_all_16
        y = y(:,:,[EDG.NULL,EDG.NOR,EDG.NCONV,EDG.NIMPL,EDG.AND] + 1); % must be before below line
    end
    if ~do_include_null
        y(:,:,1) = []; % remove null
    end
    
    y = permute(y, [1,3,2]);
    y = reshape(y, n_edges * size(y, 2), n);
end