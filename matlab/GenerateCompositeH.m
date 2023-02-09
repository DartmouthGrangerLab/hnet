% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% INPUTS
%   compbank - scalar (ComponentBank)
%   cmp - scalar (numeric index) component number
% RETURNS
%   H - n_nodes x n_nodes (int-valued numeric)
%   k - scalar (int-valued numeric)
function [H,k] = GenerateCompositeH(compbank, cmp)
    uniqEdgeType = unique(compbank.edge_states(:,cmp));
    
    n_present_edges = sum(compbank.edge_states(:,cmp) ~= EDG.NULL);
    rows = zeros(n_present_edges * 3, 1);
    cols = zeros(n_present_edges * 3, 1);
    vals = zeros(n_present_edges * 3, 1);
    
    didx = compbank.edge_endnode_idx; % store for performance

    k = 0;
    count = 0;
    for i = 1 : numel(uniqEdgeType)
        r = uniqEdgeType(i);
        if r == EDG.NULL
            continue
        end
        mask = (compbank.edge_states(:,cmp) == r);
        n_new = sum(mask);
        op = r.Op();

        edgeNodes = didx(mask,:);

        rows(count+(1:n_new)) = edgeNodes(:,1);
        cols(count+(1:n_new)) = edgeNodes(:,1);
        vals(count+(1:n_new)) = op(1); % implicit expansion
        count = count + n_new;

        rows(count+(1:n_new)) = edgeNodes(:,1);
        cols(count+(1:n_new)) = edgeNodes(:,2);
        vals(count+(1:n_new)) = op(2); % implicit expansion
        count = count + n_new;

        rows(count+(1:n_new)) = edgeNodes(:,2);
        cols(count+(1:n_new)) = edgeNodes(:,2);
        vals(count+(1:n_new)) = op(3); % implicit expansion
        count = count + n_new;

        k = k + op(4)*n_new;
    end
    
    H = sparse(rows, cols, vals, compbank.g.n_nodes, compbank.g.n_nodes);
end