% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% INPUTS
%   graphType - scalar (GRF enum)
%   n_nodes - scalar (int-valued numeric) number of nodes
% RETURNS
%   didx - n_edges x 2 (numeric index) pixel index for each edge
function didx = NeighborPairs(graphType, n_nodes)
    arguments
        graphType(1,1) GRF, n_nodes(1,1)
    end

    if graphType == GRF.GRID1D
        didx = neighbor_pairs_linear(n_nodes);
    elseif graphType == GRF.GRID2DSQR
        didx = neighbor_pairs_2d([sqrt(n_nodes),sqrt(n_nodes),1]); % for square, 1-channel images
    elseif graphType == GRF.FULL
        didx = neighbor_pairs_fully_connected(n_nodes);
    elseif graphType == GRF.SELF
        didx = neighbor_pairs_self(n_nodes);
    else
        error('unexpected graphType');
    end
end


% confirmed identical to python code
% imgSz is 3 x 1 (int-valued numeric) [n_rows,n_cols,n_chan]
function didx = neighbor_pairs_2d(imgSz)
    arguments
        imgSz (1,3)
    end
    assert(imgSz(3) == 1); % no support for multi-channel images yet

    max_edge_length = 1;
    interval = 1;

    [row,col] = PixelRowCol(imgSz);                                                                   % 1) map node_i to coord vector with dimension d, coord: N -> R^d to n-dim Euclidean
    px_coords = cat(2, row(:), col(:));
    dist = triu(squareform(pdist(px_coords)), 1);                                                     % 2) build distance matrix: take euclidean distances between all R^d pairings
    [didx1,didx2] = find((mod(round(dist), interval) == 0) & (dist <= max_edge_length) & (dist > 0)); % 3) get pixel neighbor pairs: use max_edge_length to determine neighbors
    [~,idx] = sort(didx1);
    
    didx = cat(2, didx1(:), didx2(:)); % convert from tuple of arrays to 2d array
    didx = didx(idx,:);
end


function didx = neighbor_pairs_fully_connected(n_nodes)
    didx = zeros((n_nodes*n_nodes - n_nodes) / 2, 2);
    count = 1;
    for i = 1 : n_nodes
        for j = i+1 : n_nodes
            didx(count,:) = [i,j];
            count = count + 1;
        end
    end
end


function didx = neighbor_pairs_linear(n_nodes)
    x = 1:n_nodes;
    didx = cat(2, x(1:end-1)', x(2:end)');
end


function didx = neighbor_pairs_self(n_nodes)
    didx = zeros(n_nodes, 2);
    for i = 1 : n_nodes
        didx(i,:) = [i,i];
    end
end