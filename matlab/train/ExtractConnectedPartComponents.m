% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   compbank
%   imgsz                     - 3 x 1 (int-valued numeric) [n_rows,n_cols,n_chan]
%   max_connected_part_length - scalar (numeric) max length allowed of a connected component
%   connection_thresh         - scalar (numeric) in pixels, e.g. 1.5
% RETURNS
%   newRelations
%   metadata
function [newRelations,metadata] = ExtractConnectedPartComponents(compbank, imgsz, max_connected_part_length, connection_thresh)
    arguments
        compbank(1,1) ComponentBank, imgsz(3,1), max_connected_part_length(1,1) {mustBeNumeric}, connection_thresh(1,1) {mustBeNumeric}
    end
    n_edges = compbank.g.n_edges;
    n_pts = compbank.n_cmp; % (n_images * 2) true because compbank has only gone through InitTier1EdgeRelations at this point
    
    t = tic();
    
    % split, select only the edge types we care about
    temp_relations_1 = compbank.edge_states;
    temp_relations_1(temp_relations_1 ~= EDG.NCONV) = EDG.NULL; % bit-mask each uint8 to focus node = 1; this is the NCONV relation/op, wherein the second node is active, so focus on that one
    temp_relations_2 = compbank.edge_states;
    temp_relations_2(temp_relations_2 ~= EDG.NIMPL) = EDG.NULL; % bit-mask each uint8 to focus node = 0; this is the NIMPL relation/op, wherein the first node is active, so focus on that one
    edge_relations = cat(2, temp_relations_1, temp_relations_2);
    focus_node_idx = [2.*ones(1, n_pts),ones(1, n_pts)]'; % 0010 gets a 2, 0100 gets a 1
    
    % first stab at a matlab version of the code (incomplete conversion):
    % variables to populate
    newRelations = EDG(zeros(n_edges, size(edge_relations, 2), "uint8")); % we're likely to have at least as many connected components as edge components
    srcCmpIdx = zeros(size(edge_relations, 2), 1); % we're likely to have at least as many connected components as edge components

    % precomputed values
    didx = NeighborPairs(compbank.graph_type, imgsz(1) * imgsz(2), imgsz); % n_edges x 2
    active_edge_mask = edge_relations ~= EDG.NULL; % active_edge_mask(:,i) == edges that are activated by image i
    nodeNames = strsplit(num2str(1:imgsz(1)*imgsz(2))); % cell array of chars
    
    % reused values
    activations = false(imgsz(1) * imgsz(2), 1);
    node_edges_map = cell(compbank.g.n_nodes, 1); % a node-->edges map
    
    count = 1;
    for i = 1 : n_pts % for each ~image
        % find nodes that touch edges that are active in this image (call these "active nodes")
        activeNodeIdx = didx(active_edge_mask(:,i),focus_node_idx(i)); % n_active_edges x 1 (index into nodes)
        
        % build our node-->edge map
        activeEdgeIdx = find(active_edge_mask(:,i)); % n_active_edges x 1
        node_edges_map(:) = {[]}; % empty all cells
        for j = 1 : numel(activeNodeIdx)
            node_edges_map{activeNodeIdx(j)} = [node_edges_map{activeNodeIdx(j)},activeEdgeIdx(j)];
        end

        % find pairs of active nodes that are next to each other (including on the diagonal)
        activations(:) = false;
        activations(activeNodeIdx) = true; % activeNodeIdx may contain duplicates, this will remove those
        [coordsR,coordsC] = find(reshape(activations, imgsz(1), imgsz(2))); % n_active_edges x 1 (both)
        pcoords = cat(2, coordsC(:), -coordsR(:)); % n_active_edges x 2
        dist = squareform(pdist(pcoords)); % n_active_edges x n_active_edges (aka n_pixels_that_are_white)
        [iidxR,iidxC] = find(triu(dist < connection_thresh, 1)); % n_graph_edges x 1 (both)
        g = graph(iidxR, iidxC, [], nodeNames(activations));
        
        % find strings of activated nodes that form a continously connected curve
        connectedComponents = conncomp(g, 'OutputForm', 'cell'); % 1 x n_bins (cell) node idxs in G (1-based)
        
        % create a relation for each of these connected parts
        for j = 1 : numel(connectedComponents) % for each connected part
            [newRelations,srcCmpIdx,count] = Helper(newRelations, srcCmpIdx, count, node_edges_map, edge_relations, max_connected_part_length, i, connectedComponents{j}, g);
        end
    end
    newRelations = newRelations(:,1:count-1); % in case we initialized the matrix with too many components
    srcCmpIdx = srcCmpIdx(1:count-1); % in case we initialized the matrix with too many components
    
    metadata = struct();
    src_img_idx = repmat(compbank.cmp_metadata.src_img_idx(:), 2, 1);
    metadata.src_img_idx = src_img_idx(srcCmpIdx);
    if isfield(compbank.cmp_metadata, "src_chan")
        src_chan = repmat(compbank.cmp_metadata.src_chan(:), 2, 1);
        metadata.src_chan = src_chan(srcCmpIdx);
    end
    metadata.focus_node_idx = focus_node_idx(srcCmpIdx);
    metadata.segment_idx = (1:size(newRelations, 2))';

    Toc(t);
end


function [newRelations,srcCmpIdx,count] = Helper(newRelations, srcCmpIdx, count, node_edges_map, edge_relations, max_connected_part_length, i, c, g)
    edgeidx = unique(CellCat2Vec(node_edges_map(str2double(c))));
    n_edges = numel(edgeidx);
    if n_edges < 4
        return
    elseif n_edges > max_connected_part_length
        g = minspantree(subgraph(g, c)); % find a min spanning tree of c (to remove cycles; important for below code)
        [node1,node2] = connected_components_find_central_edge(g); % find the most central node by iteratively deleting leaf nodes until only one node is left
        g = rmedge(g, node1, node2); % remove the last (~central) edge
        connected_components = conncomp(g, 'OutputForm', 'cell'); % re-get connected parts (now 2 of them)
        for j = 1 : numel(connected_components)
            [newRelations,srcCmpIdx,count] = Helper(newRelations, srcCmpIdx, count, node_edges_map, edge_relations, max_connected_part_length, i, connected_components{j}, g);
        end
    else
        assert(size(newRelations, 2) < count || all(newRelations(:,count) == EDG.NULL));
        newRelations(edgeidx,count) = edge_relations(edgeidx,i);
        srcCmpIdx(count) = i;
        count = count + 1;
    end
end


function [node1,node2] = connected_components_find_central_edge(g)
    nodeNames = table2cell(g.Nodes);
    % find the most central node by iteratively deleting leaf nodes until only one node is left
    while g.numedges > 1
        idx = find(g.degree() == 1); % get leaf nodes
        for j = 1 : numel(idx)
            if g.numedges > 1
                idx2 = g.neighbors(idx(j));
                if ~isempty(idx2)
                    g = rmedge(g, nodeNames{idx(j)}, nodeNames{idx2}); % delete the edge to this leaf (or all minus one if everything's a leaf)
                end
            end
        end
    end
    node1 = g.Edges(1,1).EndNodes{1};
    node2 = g.Edges(1,1).EndNodes{2};
end
