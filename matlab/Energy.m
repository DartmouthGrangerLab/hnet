% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% determine how well each relation matches each image
% INPUTS
%   compbank - scalar (ComponentBank)
%   data - n_nodes x n_pts (logical) list of node activations for each datapoint
% RETURNS
%   energies - n_cmp x n_pts (numeric)
function energies = Energy(compbank, data)
    arguments
        compbank(1,1) ComponentBank, data(:,:) logical
    end
    n_pts = size(data, 2);
    
    do_h_mode = true;
    
    energies = zeros(compbank.n_cmp, n_pts); % yes, double
    
    % match completely
    %   10 vs 10 = best
    %   all others = equally bad
    if do_h_mode
        data = double(data); % must be double if H is sparse (matlab technical limitation)
        
        for i = 1 : compbank.n_cmp
            [H,k] = GenerateCompositeH(compbank, i);

            energies(i,:) = dot(data, H * data) + k; % faster, identical to below
%             for j = 1 : n_pts
%                 energies(i,j) = data(:,j)' * (H * data(:,j)) + k;
%             end
        end
        energies = max(energies(:)) - energies; % convert from 0 = best to larger = better (similarity)
        % ^ makes no difference to runtime whether H is sparse or full (even when full H is single precision)
        % ^ no substantial peformance impact of transposing energies
    else
        edgeData = GetEdgeStates(data, compbank.edge_endnode_idx, compbank.edge_type_filter); % convert data to edges
        
        for i = 1 : n_pts
            energies(:,i) = sum(compbank.edge_states == edgeData(:,i) | compbank.edge_states == EDG.NULL, 1);
        end
        energies = energies - min(energies(:));
    end
    % above two versions of the code match ~exactly (within eps)
    energies = single(energies);
end