% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   compbank - scalar (ComponentBank)
%   n_nodes  - scalar (int-valued numeric) number of nodes
%   imgsz    - OPTIONAL unless graphType == GRF.GRID2D or graphType == GRF.GRID2DMULTICHAN
% RETURNS
%   didx        - n_edges x 2 (numeric index) pixel index for each edge
%   isEdgeRight - n_edges x 1 (logical)
%   isEdgeDown  - n_edges x 1 (logical)
%   nodeChan    - n_nodes x 1 (int-valued numeric)
function [didx,isEdgeRight,isEdgeDown,nodeChan] = NeighborPairs(graphType, n_nodes, imgsz)
    arguments
        graphType(1,1) GRF, n_nodes(1,1), imgsz
    end
    nodeChan = ones(n_nodes, 1);
    if graphType == GRF.GRID1D
        didx = neighbor_pairs_linear(n_nodes);
    elseif graphType == GRF.GRID2D
        didx = neighbor_pairs_2d(imgsz); % for rectangular 1-channel images
    elseif graphType == GRF.GRID2DMULTICHAN
        [didx,nodeChan] = neighbor_pairs_2d_multichan(imgsz); % for rectangular n-channel images
    elseif graphType == GRF.FULL
        didx = neighbor_pairs_fully_connected(n_nodes);
    elseif graphType == GRF.SELF
        didx = neighbor_pairs_self(n_nodes);
    else
        error("unexpected graphType");
    end
    
    if graphType == GRF.GRID2D || graphType == GRF.GRID2DMULTICHAN
        [row,col] = PixelRowCol(imgsz);
        isEdgeRight = col(didx(:,1)) ~= col(didx(:,2));
        isEdgeDown = row(didx(:,1)) ~= row(didx(:,2));
    else
        isEdgeRight = false(size(didx, 1), 1);
        isEdgeDown = false(size(didx, 1), 1);
    end
end


% confirmed identical to python code
% imgsz is 3 x 1 (int-valued numeric) [n_rows,n_cols,n_chan]
function didx = neighbor_pairs_2d(imgsz)
    arguments
        imgsz(3,1)
    end
    assert(imgsz(3) == 1); % no support for multi-channel images here

    max_edge_length = 1;
    interval = 1;

    [row,col] = PixelRowCol(imgsz);                                                                   % 1) map node_i to coord vector with dimension d, coord: N -> R^d to n-dim Euclidean
    px_coords = cat(2, row(:), col(:));
    dist = triu(squareform(pdist(px_coords)), 1);                                                     % 2) build distance matrix: take euclidean distances between all R^d pairings
    [didx1,didx2] = find((mod(round(dist), interval) == 0) & (dist <= max_edge_length) & (dist > 0)); % 3) get pixel neighbor pairs: use max_edge_length to determine neighbors
    [~,idx] = sort(didx1);
    
    didx = cat(2, didx1(:), didx2(:)); % convert from tuple of arrays to 2d array
    didx = didx(idx,:);
end


function [didx,nodeChan] = neighbor_pairs_2d_multichan(imgsz)
    arguments
        imgsz(3,1)
    end
    assert(imgsz(3) > 1);
    
    didx = neighbor_pairs_2d([imgsz(1),imgsz(2),1]);
    
    % replicate across channels
    % didx actual values must change (the number of nodes increases)
    n_nodes_per_chan = imgsz(1) * imgsz(2);
    n_edges_per_chan = size(didx, 1);
    didx = repmat(didx, [imgsz(3),1]);
    nodeChan = ones(n_nodes_per_chan*imgsz(3), 1);
    for i = 2 : imgsz(3) % for each channel > 1
        didx((i-1)*n_edges_per_chan + (1:n_edges_per_chan),:) = didx((i-1)*n_edges_per_chan + (1:n_edges_per_chan),:) + n_nodes_per_chan;
        nodeChan((i-1)*n_nodes_per_chan + (1:n_nodes_per_chan)) = i;
    end
    
    % NOT connecting across channels
end


function didx = neighbor_pairs_fully_connected(n_nodes)
    arguments
        n_nodes(1,1)
    end
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
    arguments
        n_nodes(1,1)
    end
    x = 1:n_nodes;
    didx = cat(2, x(1:end-1)', x(2:end)');
end


function didx = neighbor_pairs_self(n_nodes)
    arguments
        n_nodes(1,1)
    end
    didx = zeros(n_nodes, 2);
    for i = 1 : n_nodes
        didx(i,:) = [i,i];
    end
end