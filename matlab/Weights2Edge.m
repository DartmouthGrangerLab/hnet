% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   data - n_edges*EDG.n x n (numeric) edges, as returned by GetRelations
%   do_include_na - OPTIONAL scalar (logical) (default = true)
% RETURNS
%   y - n_edges x n (EDG enum)
function y = Weights2Edge(x, do_include_na)
    if ~exist("do_include_na", "var") || isempty(do_include_na)
        do_include_na = true;
    end
    n = size(x, 2);
    
    if do_include_na
        x = reshape(x, [], EDG.n, n); % n_edges x n_uniq_edg x n
    else
        x = reshape(x, [], EDG.n - 1, n); % n_edges x n_uniq_edg x n
    end
    
    x = permute(x, [2,1,3]); % now EDG.n x n_edges x n
    n_edges = size(x, 2);
    
    y = EDG(zeros(n_edges, n));
    
    for i = 1 : n
        [val,idx] = max(abs(x(:,:,i)), [], 1);
        if do_include_na
            y(:,i) = EDG(idx-1); % -1 because idx is 1-->17, we want 0-->16
        else
            idx(val==0) = 0; % if no max, it's EDG.NULL
            y(:,i) = EDG(idx); % no -1, because data doesn't contain EDG.NULL (aka 0)
        end
    end
end