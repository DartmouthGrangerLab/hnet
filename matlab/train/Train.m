% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   cfg    - scalar (struct) configuration struct
%   layout
%   dat    - scalar (Dataset) training dataset
% RETURNS
%   model
function model = Train(cfg, layout, dat)
    arguments
        cfg(1,1) struct, layout(:,1) cell, dat(1,1) Dataset
    end
    
    SetRNG(cfg);
    
    model = Model(layout, dat.n_nodes, dat.n_classes, dat.node_name);
    
    steps = strsplit(cfg.trn_spec, '-->');
    for ii = 1 : numel(steps)
        step = strsplit(steps{ii}, '.'); % <bank>.<task>.<taskparams>
        bank = step{1};
        task = step{2};

        if strcmp(task, 'memorize')
            model = InsertComponents(model, bank, dat.n_pts);
            model.compbanks.(bank).edge_states(:,end-dat.n_pts+1:end) = GetEdgeStates(dat.pixels, model.compbanks.(bank).edge_endnode_idx, model.compbanks.(bank).edge_type_filter); % convert from pixels to edges
            model.compbanks.(bank).meta = struct(src_img_idx=1:dat.n_pts);
        elseif strcmp(task, 'extractconnec')
            max_length = str2double(step{3}); % max length of a connected component (e.g. Inf, 20)
            model = ExtractConnectedPartComponents(model, bank, dat.img_sz, max_length, 1.5);
        elseif strcmp(task, 'transl') % translate
            max_translation_delta = str2double(step{3}); % subscript is an integer representing number of pixels to translate
            max_rot = 0; % max rotation in degrees
            model = TranslateAndRotate(model, bank, dat.img_sz, max_translation_delta, max_rot);
        elseif strcmp(task, 'extractcorr') % extract components from input data based on correlation across datapoints
            clusterer = step{3}; % 'kmeans', 'gmm', 'hierarchical<linkage>', 'spectralkmeans<1|2|3>', 'ica'
            k = str2double(step{4});
            max_edges_per_cmp = str2double(step{5});
            mode = step{6};
            [~,inBanks] = inedges(model.g, bank);
            model = FactorEdgesToExtractComponents(model, dat, clusterer, k, max_edges_per_cmp, inBanks{1}, bank, mode);
        else
            error('unexpected task');
        end
    end
end