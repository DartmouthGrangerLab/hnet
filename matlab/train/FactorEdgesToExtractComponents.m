% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   model
%   dat
%   alg               - (char) 'ica' | ... more pending ... | 'spectralkmeans' | 'kmeans' | 'gmm' | 'hierarchical*'
%   k                 - scalar (int-valued numeric)
%   max_edges_per_cmp - scalar (int-valued numeric)
%   bank2Cluster      - (char)
%   bank2FormCmp      - (char)
%   mode              - (char) 'unsup' | 'sup1' | 'sup2' | 'unsupsplit' | 'sup1split' | 'sup2split'
% RETURNS
%   model
function model = FactorEdgesToExtractComponents(model, dat, alg, k, max_edges_per_cmp, bank2Cluster, bank2FormCmp, mode)
    arguments
        model(1,1) Model, dat(1,1) Dataset, alg(1,:) char, k(1,1), max_edges_per_cmp(1,1), bank2Cluster(1,:) char, bank2FormCmp(1,:) char, mode(1,:) char
    end
    k_per_class = round(k / dat.n_classes);
    n_edges = model.compbanks.(bank2FormCmp).g.n_edges;

    t = tic();
    
    compCode = Encode(model, dat.pixels);
    nodeActivations = compCode.(bank2Cluster); % n_nodes x n_pts (logical or numeric)
    n_pts = size(nodeActivations, 2);
    
    if startsWith(mode, "sup1") || startsWith(mode, "sup2")
        assert(dat.n_classes == 2);
        nodeActivations(end+1,:) = logical(dat.label_idx-1);
        didx = NeighborPairs(model.compbanks.(bank2FormCmp).graph_type, model.compbanks.(bank2FormCmp).n_nodes+1);
    else
        didx = model.compbanks.(bank2FormCmp).edge_endnode_idx;
    end
    
    edgeRelations = GetEdgeStates(nodeActivations, didx, model.compbanks.(bank2FormCmp).edge_type_filter);
    
    if endsWith(mode, "split")
        edgeStates = EDG([]);
        for c = 1 : dat.n_classes
            currEdgeRelations = edgeRelations(:,dat.label_idx == c);
            
            usedEdgeMsk = (sum(currEdgeRelations, 2) > n_pts * 0.05); % ignore edges that are almost always n/a
        
            curr = EDG(zeros(n_edges, k_per_class, "uint8"));
            curr(usedEdgeMsk,:) = Factor(alg, currEdgeRelations(usedEdgeMsk,:), k_per_class, model.compbanks.(bank2FormCmp).edge_type_filter);
            if startsWith(mode, "sup1") % if sup2, leave them in for the below func
                curr(didx(:,1) == max(didx(:)) | didx(:,2) == max(didx(:)),:) = EDG.NULL; % remove edges to the dv (last node)
            end
            curr(usedEdgeMsk,:) = CropLeastCoOccurringEdges(curr(usedEdgeMsk,:), currEdgeRelations(usedEdgeMsk,:), max_edges_per_cmp);
            if startsWith(mode, "sup2")
                curr(didx(:,1) == max(didx(:)) | didx(:,2) == max(didx(:)),:) = EDG.NULL; % remove edges to the dv (last node)
            end
            
            edgeStates = cat(2, edgeStates, curr);
        end
    else
        usedEdgeMsk = (sum(edgeRelations, 2) > n_pts * 0.05); % ignore edges that are almost always n/a
        
        edgeStates = EDG(zeros(n_edges, k, "uint8"));
        edgeStates(usedEdgeMsk,:) = Factor(alg, edgeRelations(usedEdgeMsk,:), k, model.compbanks.(bank2FormCmp).edge_type_filter);
        if startsWith(mode, "sup1") % if sup2, leave them in for the below func
            edgeStates(didx(:,1) == max(didx(:)) | didx(:,2) == max(didx(:)),:) = EDG.NULL; % remove edges to the dv (last node)
        end
        edgeStates(usedEdgeMsk,:) = CropLeastCoOccurringEdges(edgeStates(usedEdgeMsk,:), edgeRelations(usedEdgeMsk,:), max_edges_per_cmp);
        if startsWith(mode, "sup2")
            edgeStates(didx(:,1) == max(didx(:)) | didx(:,2) == max(didx(:)),:) = EDG.NULL; % remove edges to the dv (last node)
        end
    end
    
    if startsWith(mode, "sup1") || startsWith(mode, "sup2")
        edgeStates(didx(:,1) == max(didx(:)) | didx(:,2) == max(didx(:)),:) = [];
        didx(didx(:,1) == max(didx(:)) | didx(:,2) == max(didx(:)),:) = [];
        assert(all(didx(:) == model.compbanks.(bank2FormCmp).edge_endnode_idx(:)));
    end
    
    % handle too few edges
    nEdgesPerCmp = sum(edgeStates ~= EDG.NULL, 1);
    mask = (nEdgesPerCmp < Config.MIN_EDGES_PER_CMP);
    disp("removing " + num2str(sum(mask)) + " components for having too few edges");
    edgeStates(:,mask) = [];

    model = model.ClearComponents(bank2FormCmp);
    model = model.InsertComponents(bank2FormCmp, size(edgeStates, 2));
    model = model.SetEdgeStates(bank2FormCmp, edgeStates);

    Toc(t, toc(t) > 1);
end


% INPUTS
%   alg            - (char)
%   edges          - n_edges x n_pts (EDG enum)
%   k              - scalar (int-valued numeric)
%   edgeTypeFilter - vector (EDG enum)
function relations = Factor(alg, edges, k, edgeTypeFilter)
    assert(k > 1);
    [n_edges,n_pts] = size(edges);
    
    if startsWith(alg, "ica")
        if alg == "ica"
            icaComponents = fastica(double(Edge2Logical(edges, true))', 'numOfIC', k, 'verbose', 'off');
            relations = Weights2Edge(icaComponents', true);
        elseif alg == "icacrop"
            icaComponents = fastica(double(Edge2Logical(edges, false))', 'numOfIC', k, 'verbose', 'off');
            temp = abs(icaComponents(icaComponents~=0));
            icaComponents(abs(icaComponents) < quantile(temp, 0.25)) = 0; % remove small non-0 weights
            relations = Weights2Edge(icaComponents', false);
        elseif alg == "icacroplots"
            icaComponents = fastica(double(Edge2Logical(edges, false))', 'numOfIC', k, 'verbose', 'off');
            temp = abs(icaComponents(icaComponents~=0));
            icaComponents(abs(icaComponents) < quantile(temp, 0.75)) = 0; % remove small non-0 weights
            relations = Weights2Edge(icaComponents', false);
        elseif alg == "icacropsome"
            icaComponents = fastica(double(Edge2Logical(edges, false))', 'numOfIC', k, 'verbose', 'off');
            temp = abs(icaComponents(icaComponents~=0));
            icaComponents(abs(icaComponents) < quantile(temp, 0.5)) = 0; % remove small non-0 weights
            relations = Weights2Edge(icaComponents', false);
        else
            error("unexpected alg");
        end
        relations = FilterEdgeType(relations, edgeTypeFilter);
    else
        data = double(edges);
        distance = 'cosine';
        if numel(unique(edges(:))) ~= 2 % if we have multiple kinds of edges (vs just NA and AND), cosine distance won't work until we convert to one-hot
            % jaccard = convert to one-hot, then use cosine
            D = size(edges, 2); % store now, it'll change below
            data = double(edges);
            for i = 1 : D
                data = cat(2, data, encode.OneHot(data(:,i)));
            end
            data(:,1:D) = []; % remove the originals
        end
        
        if startsWith(alg, "spectralkmeans")
            data = 1 - squareform(pdist(data, distance)); % n_edges x n_edges (double) pairwise similarity
            clustMdl = ml.ClustInit(char(alg), k, 'euclidean', 'kmeans++');
        else
            clustMdl = ml.ClustInit(char(alg), k, distance, 'random', data);
        end
        [~,~,idx] = ml.Cluster(clustMdl, data, false, 25);
        
        % convert back to EDG relations
        relations = EDG(encode.OneHot(idx, k)); % n_edges x k;
        counts = zeros(EDG.n, n_edges);
        for i = 1 : n_edges
            counts(:,i) = CountNumericOccurrences(double(edges(i,:))+1, 1:EDG.n); % for edge temp(j), across datapoints, how often do we see each edge type?
        end
        counts(1,:) = 0; % we don't want to choose EDG.NULL if we can help it
        for i = 1 : n_edges
            if any(counts(:,i))
                [~,maxEdge] = max(counts(:,i));
                relations(i,idx(i)) = EDG(maxEdge - 1); % EDG is {0 .. 16}, maxEdge is {1 .. 17}
            else
                relations(i,idx(i)) = EDG.NULL;
            end
        end
    end
end


% handle too many edges by removing edges that co-occur least often with other edges of the component
function relations = CropLeastCoOccurringEdges(relations, edges, max_edges_per_cmp)
    nEdgesPerCmp = sum(relations ~= EDG.NULL, 1); % 1 x n_cmp
    nToRemove = nEdgesPerCmp - max_edges_per_cmp; % 1 x n_cmp
    disp("cropping " + num2str(sum(nToRemove > 0)) + " components for having " + num2str(sum(nToRemove(nToRemove > 0))) + " too many edges in total");

    if IsJuliaConfigured() && any(nToRemove > 0)
        relations = EDG(Julia(fullfile(fileparts(mfilename('fullpath')), '..', 'julia_code.jl'), 'crop_least_co_occurring_edges', uint8(relations), uint8(edges), nToRemove(:)));
        % above confirmed identical to below, 2x as fast (for long runs) but requires a bunch of julia setup
    else
        cmpIdx = find(nToRemove > 0);
        for i = cmpIdx
            while nToRemove(i) > 0
                relationIdx = find(relations(:,i)); % indices of all the non-0 edges

                code = (edges(relationIdx,:) == relations(relationIdx,i));

                pDist = squareform(pdist(single(code), "cosine")); % 1 minus cosine similarity
                [~,idx] = max(sum(pDist, 2));

                relations(relationIdx(idx),i) = EDG.NULL;
                nToRemove(i) = nToRemove(i) - 1;
            end
        end
    end
end