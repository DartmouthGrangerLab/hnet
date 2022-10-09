% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   model
%   bank                      - (char) name of component bank to operate on
%   imgSz                     - 3 x 1 (int-valued numeric) [n_rows,n_cols,n_chan]
%   max_connected_part_length - scalar (numeric) max length allowed of a connected component
%   connection_thresh         - scalar (numeric) in pixels, e.g. 1.5
% RETURNS
%   model
function model = ExtractConnectedPartComponents(model, bank, imgSz, max_connected_part_length, connection_thresh)
    arguments
        model                     (1,1) Model
        bank                      (1,:) char
        imgSz                     (3,1)
        max_connected_part_length (1,1) {mustBeNumeric}
        connection_thresh         (1,1) {mustBeNumeric}
    end
    n_edges = model.compbanks.(bank).n_edges;
    n_pts = model.compbanks.(bank).n_cmp; % (n_images * 2) true because compbank has only gone through InitTier1EdgeRelations at this point
    
    t = tic();
    
    % split, select only the edge types we care about
    temp_relations_1 = model.compbanks.(bank).edge_states;
    temp_relations_1(temp_relations_1 ~= EDG.NCONV) = EDG.NULL; % bit-mask each uint8 to focus node = 1; this is the NCONV relation/op, wherein the second node is active, so focus on that one
    temp_relations_2 = model.compbanks.(bank).edge_states;
    temp_relations_2(temp_relations_2 ~= EDG.NIMPL) = EDG.NULL; % bit-mask each uint8 to focus node = 0; this is the NIMPL relation/op, wherein the first node is active, so focus on that one
    edge_relations = cat(2, temp_relations_1, temp_relations_2);
    metadata = struct(src_img_idx=[1:n_pts,1:n_pts], focus_node_idx=[2.*ones(1, n_pts),ones(1, n_pts)]); % 0010 gets a 2, 0100 gets a 1
    
    % first stab at a matlab version of the code (incomplete conversion):
    % variables to populate
    newRelations = EDG(zeros(n_edges, size(edge_relations, 2), 'uint8')); % we're likely to have at least as many connected components as edge components
    srcCmpIdx = zeros(size(edge_relations, 2), 1); % we're likely to have at least as many connected components as edge components

    % precomputed values
    px_didx = NeighborPairs(GRF.GRID2DSQR, imgSz(1) * imgSz(2));
    active_rels = (edge_relations ~= EDG.NULL); % edges that are activated by image i
    nodeNames = strsplit(num2str(1:imgSz(1)*imgSz(2))); % cell array of chars
    
    % reused values
    activations = false(imgSz(1) * imgSz(2), 1);
    zzz = cell(1, model.compbanks.(bank).n_nodes); % a node-->edges map
    
    count = 1;
    for i = 1 : n_pts % for each ~image
        % find nodes that touch edges that are active in this image (call these "active nodes")
        activeNodeIdx = px_didx(active_rels(:,i),metadata.focus_node_idx(i)); % n_active_edges x 1 (index into nodes)
        
        % build our node-->edge map
        activeRelIdx = find(active_rels(:,i)); % n_active_edges x 1
        zzz(:) = {[]}; % empty all cells
        for j = 1 : numel(activeNodeIdx)
            zzz{activeNodeIdx(j)} = [zzz{activeNodeIdx(j)},activeRelIdx(j)];
        end

        % find pairs of active nodes that are next to each other (including on the diagonal)
        activations(:) = false;
        activations(activeNodeIdx) = true; % activeNodeIdx may contain duplicates, this will remove those
        [coordsR,coordsC] = find(reshape(activations, imgSz(1), imgSz(2))); % n_activeNodeIdx x 1 (both)
        pcoords = cat(2, coordsC(:), -coordsR(:)); % n_activeNodeIdx x 2
        dist = squareform(pdist(pcoords)); % n_activeNodeIdx x n_activeNodeIdx (aka n_pixels_that_are_white)
        [iidxR,iidxC] = find(triu(dist < connection_thresh, 1)); % n_graph_edges x 1 (both)
        g = graph(iidxR, iidxC, [], nodeNames(activations));
        
        % find strings of activated nodes that form a continously connected curve
        connectedComponents = conncomp(g, 'OutputForm', 'cell'); % 1 x n_bins (cell) node idxs in G (1-based)
        
        % create a relation for each of these connected parts
        for j = 1 : numel(connectedComponents) % for each connected part
            [newRelations,srcCmpIdx,count] = Helper(newRelations, srcCmpIdx, count, zzz, edge_relations, max_connected_part_length, i, connectedComponents{j}, g);
        end
    end
    newRelations = newRelations(:,1:count-1); % in case we initialized the matrix with too many components
    srcCmpIdx = srcCmpIdx(1:count-1); % in case we initialized the matrix with too many components
    
    newMeta = struct();
    newMeta.src_img_idx = metadata.src_img_idx(srcCmpIdx);
    newMeta.focus_node_idx = metadata.focus_node_idx(srcCmpIdx);
    newMeta.segment_idx = 1:size(newRelations, 2);

    model = ClearComponents(model, bank);
    model = InsertComponents(model, bank, size(newRelations, 2));
    model.compbanks.(bank).edge_states(:) = newRelations;
    model.compbanks.(bank).meta = newMeta;

    Toc(t);
end


function [newRelations,srcCmpIdx,count] = Helper(newRelations, srcCmpIdx, count, zzz, edge_relations, max_connected_part_length, i, c, g)
    n_edges = numel(unique(CellCat2Vec(zzz(str2double(c)))));
    if n_edges < 4
        return
    elseif n_edges > max_connected_part_length
        g = minspantree(subgraph(g, c)); % find a min spanning tree of c (to remove cycles; important for below code)
        [node1,node2] = connected_components_find_central_edge(g); % find the most central node by iteratively deleting leaf nodes until only one node is left
        g = rmedge(g, node1, node2); % remove the last (~central) edge
        connectedComponents = conncomp(g, 'OutputForm', 'cell'); % re-get connected parts (now 2 of them)
        for j = 1 : numel(connectedComponents)
            [newRelations,srcCmpIdx,count] = Helper(newRelations, srcCmpIdx, count, zzz, edge_relations, max_connected_part_length, i, connectedComponents{j}, g);
        end
    else
        newRelations(:,count) = EDG.NULL;
        for j = 1 : numel(c) % for each node in this connected part
            newRelations(zzz{str2double(c{j})},count) = edge_relations(zzz{str2double(c{j})},i);
        end
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
