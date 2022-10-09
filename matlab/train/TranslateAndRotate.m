% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% you will notice slight differences in results between this code and the python code; this one is more correct (python code drops a couple edges)
% INPUTS
%   model
%   bank                  - (char) bank to translate / rotate
%   imgSz                 - 3 x 1 (int-valued numeric) [n_rows,n_cols,n_chan]
%   max_translation_delta - scalar (numeric)
%   max_rot               - scalar (numeric)
% RETURNS
%   model
function model = TranslateAndRotate(model, bank, imgSz, max_translation_delta, max_rot)
    arguments
        model(1,1) Model, bank(1,:) char, imgSz(3,1), max_translation_delta(1,1), max_rot(1,1)
    end
    
    compbank = model.compbanks.(bank);
    n_edges = compbank.n_edges;
    n = compbank.n_cmp; % number of input components
    
    px_didx = compbank.edge_endnode_idx; % n_edges x 2 (numeric idx)
    
    t = tic();
    
    compbank.meta.pretranslate_idx = 1:compbank.n_cmp;
    
    th_int = 45; % theta interval in degrees
    delta = -max_translation_delta:max_translation_delta;
    theta = -max_rot:th_int:max_rot;
    n_translations = numel(delta) * numel(delta) * numel(theta) - 1; % was = np.max(coords[:,0]) + 1

    nodesToEdge = -1 .* ones(compbank.n_nodes, compbank.n_nodes);
    for i = 1 : n_edges
        nodesToEdge(px_didx(i,1),px_didx(i,2)) = i;
    end

    fn = {'src_img_idx','focus_node','segment_idx','group_idx','pretranslate_idx'};
    fn = fn(isfield(compbank.meta, fn)); % subset fields to only those present
    
    edgeStates = EDG(zeros(compbank.n_edges, n, n_translations));
    metadata = struct();
    for i = 1 : numel(fn)
        if isfield(compbank.meta, fn{i})
            metadata.(fn{i}) = zeros(n, n_translations, 'like', compbank.meta.(fn{i})); % create a new dictionary with the same keys but empty lists for values
        end
    end
    metadata.translation_idx = zeros(n, n_translations);
    metadata.offset_x        = zeros(n, n_translations);
    metadata.offset_y        = zeros(n, n_translations);
    metadata.degrees         = zeros(n, n_translations);

    count = 1; % translation number/id/index
    for offset_x = delta
        for offset_y = delta
            for degrees = theta
                if offset_x == 0 && offset_y == 0 && degrees == 0 % not translated
                    continue
                end
                % begin code for a single translation

                % compute the node index for each translated edge
                J = geom.TranslationMat(offset_y, offset_x) * geom.RotationMat(-degrees, 'deg', true); % no idea why x and y are inverted, degrees is negative
                [row,col] = find(PermutationMat(J, imgSz(1))); % n_pixels x n_pixels
                permute_node_to_node = -1 .* ones(imgSz(1) * imgSz(2), 1); % n_nodes x 1
                permute_node_to_node(row) = col;
                node_from = permute_node_to_node(px_didx(:,1)); % n_edges x 1
                node_to = permute_node_to_node(px_didx(:,2)); % n_edges x 1
                new_edge = zeros(n_edges, 1);
                for j = 1 : n_edges
                    if node_from(j) > 0 && node_to(j) > 0
                        if node_from(j) > node_to(j)
                            new_edge(j) = nodesToEdge(node_to(j),node_from(j)); % swap
                        else
                            new_edge(j) = nodesToEdge(node_from(j),node_to(j));
                        end
                    end
                end
                valid_edge_mask = (new_edge > 0); % n_edges x 1
                
                edges_from = find(valid_edge_mask); % n_valid_edges x 1
                edges_to = new_edge(valid_edge_mask); % n_valid_edges x 1

                assert(all(all(edgeStates(:,:,count) == EDG.NULL)));
                edgeStates(edges_to,:,count) = compbank.edge_states(edges_from,:);
                metadata.translation_idx(:,count) = 1 + count; % translation number/id/index (1 = not translated)
                metadata.offset_x(:,count)        = offset_x;
                metadata.offset_y(:,count)        = offset_y;
                metadata.degrees(:,count)         = degrees;
                for j = 1 : numel(fn)
                    metadata.(fn{j})(:,count) = compbank.meta.(fn{j});
                end
                count = count + 1; % translation number/id/index
            end
        end
    end
    
    edgeStates = reshape(edgeStates, size(edgeStates, 1), size(edgeStates, 2) * size(edgeStates, 3));
    edgeStates = cat(2, compbank.edge_states, edgeStates);
    
    fn = fieldnames(metadata);
    for i = 1 : numel(fn)
        metadata.(fn{i}) = metadata.(fn{i})(:)';
        if strcmp(fn{i}, 'translation_idx')
            metadata.(fn{i}) = cat(2, ones(1, compbank.n_cmp), metadata.(fn{i}));
        elseif strcmp(fn{i}, 'offset_x') || strcmp(fn{i}, 'offset_y') || strcmp(fn{i}, 'degrees')
            metadata.(fn{i}) = cat(2, zeros(1, compbank.n_cmp), metadata.(fn{i}));
        else
            metadata.(fn{i}) = cat(2, compbank.meta.(fn{i}), metadata.(fn{i}));
        end
    end
    
    if isfield(model.compbanks, 'group')
        if model.compbanks.group.n_cmp > 0
            groupIdx = GroupIdx(model, 'group'); % MUST be before we change the model
        else % groups not yet initialized, so we'll just group by input component number
            groupIdx = 1:n;
        end
        groupIdx = repmat(groupIdx(:), n_translations + 1, 1); % n * n_translations+1 x 1 (+1 for the original non-translated copies)
    end
    
    % update component bank
    model = ClearComponents(model, bank);
    model = InsertComponents(model, bank, size(edgeStates, 2));
    model.compbanks.(bank).edge_states(:) = edgeStates;
    model.compbanks.(bank).meta = metadata;
    
    % update groups
    if isfield(model.compbanks, 'group')
        assert(model.compbanks.group.n_cmp == 0);
        model = InsertComponents(model, 'group', numel(unique(groupIdx)));
        model.compbanks.group.edge_states(:) = EDG.NULL;
        for i = 1 : model.compbanks.group.n_cmp
            model.compbanks.group.edge_states(groupIdx==i,i) = EDG.AND;
        end

        [~,grpIn] = inedges(model.g, 'group');
        model.compbanks.group.meta = model.compbanks.(grpIn{1}).meta;
    end
    
    Toc(t);
end