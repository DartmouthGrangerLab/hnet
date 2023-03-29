% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   modelName    - (char) name of a model (see Layout.m)
%   frontendSpec - (char) dataset and frontend name and parameters (see dataset.m)
%   trnSpec      - (char) training specification string
% USAGE
%   Main('metacred',  'ucicreditgerman', 'tier1.memorize-->tier1.extractcorr.icacropsome.100.50.unsupsplit-->meta.extractcorr.kmeans.10.50.unsupsplit');
%   Main('groupedimg', 'mnistpy.128', 'connectedpart.memorize-->connectedpart.extractconnec.25-->connectedpart.transl.2');
function [] = Main(modelName, frontendSpec, trnSpec)
    addpath(genpath(Config.MyDir())); % add sub-folders in case they weren't added
    
    cfg = struct(model_name=modelName, frontend_spec=frontendSpec, trn_spec=trnSpec);
    outDir = fullfile(Config.OUT_DIR, [cfg.model_name,'_',cfg.frontend_spec,'_',strrep(cfg.trn_spec, '-->', '-')]);
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    disp(['-- ',outDir,' --']);
    
    layout = Layout(modelName);
    
    %% load dataset
    SetRNG(1000);
    trnDataset = Dataset(cfg.frontend_spec, 'trn');
    tstDataset = Dataset(cfg.frontend_spec, 'tst');
    
    % print dataset info to text files
    temp = cat(2, trnDataset.node_name, num2cell(sum(trnDataset.pixels, 2)));
    writecell(cat(1, {'node_name','num nonzero pixels'}, temp), fullfile(Config.OUT_DIR, ['node_name_',cfg.frontend_spec,'.csv']));
    if isfield(trnDataset.meta, "category_info")
        temp = cat(2, fieldnames(trnDataset.meta.category_info), struct2cell(trnDataset.meta.category_info));
        writecell(cat(1, {'field','value'}, temp), fullfile(Config.OUT_DIR, ['category_info_',cfg.frontend_spec,'.csv']));
    end
    if isfield(trnDataset.meta, "t")
        writetable(trnDataset.meta.t, fullfile(Config.OUT_DIR, [cfg.frontend_spec,'_trn.csv']));
        writetable(tstDataset.meta.t, fullfile(Config.OUT_DIR, [cfg.frontend_spec,'_tst.csv']));
    end
    if isfield(trnDataset.meta, "t_bin")
        writetable(trnDataset.meta.t_bin, fullfile(Config.OUT_DIR, [cfg.frontend_spec,'_bin_trn.csv']));
        writetable(tstDataset.meta.t_bin, fullfile(Config.OUT_DIR, [cfg.frontend_spec,'_bin_tst.csv']));
    end
    t = table(trnDataset.node_name(:));
    for i = 1 : trnDataset.n_classes
        t = addvars(t, round(sum(trnDataset.pixels(:,trnDataset.label_idx == i), 2) ./ trnDataset.n_pts, 4), 'NewVariableNames', ['frac_occurs_in_pts_of_class_',trnDataset.uniq_classes{i}]);
    end
    writetable(t, fullfile(Config.OUT_DIR, [cfg.frontend_spec,'_var_cls_cooccur_frequency.csv']));
    
    % render example datapoints
    for c = 1 : trnDataset.n_classes
        idx = find(trnDataset.label_idx == c);
        for i = 1 : min(numel(idx), 5) % for each image in this class
            RenderDatapointNoEdges(fullfile(Config.OUT_DIR, 'samplestim'), trnDataset, idx(i), [trnDataset.uniq_classes{c},'_',num2str(i)])
        end
    end
    
    %% train
    if Config.DO_CACHE
        model = CachedCompute(@Train, cfg, layout, trnDataset); % with caching
    else
        model = Train(cfg, layout, trnDataset); % without caching
    end
    
    SetRNG(cfg); % reset RNG after training
    
    %% encode
    trnCode = struct();
    tstCode = struct();

    [trnCode.comp_code,trnCode.premerge_idx] = Encode(model, trnDataset);
    [tstCode.comp_code,tstCode.premerge_idx] = Encode(model, tstDataset);

    trnCode.hist = struct();
    tstCode.hist = struct();
    trnCode.comp_best_img = struct();
    tstCode.comp_best_img = struct();

    for i = 1 : model.n_compbanks
        name = model.compbank_names{i};

        trnCode.hist.(name) = ClassHistogram(trnDataset, trnCode.comp_code.(name));
        tstCode.hist.(name) = ClassHistogram(tstDataset, tstCode.comp_code.(name));

        [~,trnCode.comp_best_img.(name)] = max(trnCode.comp_code.(name), [], 2);
        [~,tstCode.comp_best_img.(name)] = max(tstCode.comp_code.(name), [], 2);
    end

    %% print accuracy and related stats
    PrintPerformance(model, trnDataset, tstDataset, trnCode.comp_code.(model.output_bank_name), tstCode.comp_code.(model.output_bank_name), [cfg.model_name,'_',cfg.trn_spec], cfg.frontend_spec);
    
    %% render the edges involved in each component
    for i = 1 : model.n_compbanks
        bank = model.compbank_names{i};
        
        RenderAll(outDir, model, tstDataset, tstCode, bank, true, true);
        RenderAll(outDir, model, tstDataset, tstCode, bank, true, false);
        RenderAll(outDir, model, tstDataset, tstCode, bank, false, true);

        % get metadata and pass to PrintEdgeRelations
        node_info = model.compbanks.(bank).g.node_metadata.name;
        if strcmp(bank, model.tier1_compbank_names{1}) && isfield(tstDataset.meta, "category_info")
            for j = 1 : tstDataset.n_nodes
                if isfield(tstDataset.meta.category_info, node_info{j})
                    node_info{j} = tstDataset.meta.category_info.(node_info{j});
                end
            end
        end

        for j = 1 : min(100, model.compbanks.(bank).n_cmp) % for each component in this bank
            PrintEdgeRelations(outDir, [bank,'_cmp',int2str(j),'.txt'], model.compbanks.(bank).edge_states(:,j), model.compbanks.(bank).edge_endnode_idx, node_info, bank, model.tier1_compbank_names{1});
        end
    end

    %% explain each tst datapoint
    RenderTstToExplain(outDir, model, tstDataset, tstCode, model.output_bank_name, true);
    RenderTstToExplain(outDir, model, tstDataset, tstCode, model.output_bank_name, false);
    
    %% render component best matches
    for i = 1 : numel(model.tier1_compbank_names)
        % can only call for tier 1 and tier 2 component banks
        RenderComponentBestMatches(outDir, model, tstDataset, tstCode, model.tier1_compbank_names{i});
        [~,downstreamBankNames] = outedges(model.g, model.tier1_compbank_names{i});
        for j = 1 : numel(downstreamBankNames)
            if ~strcmp(downstreamBankNames{j}, 'out')
                RenderComponentBestMatches(outDir, model, tstDataset, tstCode, downstreamBankNames{j});
            end
        end
    end
    
    %% render class histograms
    for i = 1 : model.n_compbanks
        RenderHist(outDir, trnDataset.uniq_classes, trnCode, model.compbank_names{i}, 'trn');
        RenderHist(outDir, trnDataset.uniq_classes, tstCode, model.compbank_names{i}, 'tst');
    end

    %% render discriminability vs sharedness of each component's response to the dataset
    if ~isfield(model.compbanks, "meta") && ~isfield(model.compbanks, "group") && trnDataset.n_classes > 2 % if we only have one tier
        RenderDiscrimVsSharednessVsFrequency(outDir, model, trnCode, trnDataset, 'trn');
        RenderDiscrimVsSharednessVsFrequency(outDir, model, tstCode, tstDataset, 'tst');
    end
    
    %% render the groups
    if isfield(model.compbanks, "group")
        for i = 1 : min(10, model.compbanks.group.n_cmp) % for each group
            RenderGroup(outDir, model, tstDataset, tstCode, i);
        end
    end
    
    %% precompute some stats for below figures
    if islogical(trnCode.comp_code.(model.output_bank_name)) || isa(trnCode.comp_code.(model.output_bank_name), 'uint8')
        knnParams = struct(k=1, distance='dot');
        nbParams = struct(distribution='bern');
    else % feat codes are probably energies; ~poisson distributed
        knnParams = struct(k=1, distance='cosine');
        nbParams = struct(distribution='gauss');
    end
    trnSenseEdges = Edge2Logical(GetEdgeStates(trnDataset.pixels, model.compbanks.(model.tier1_compbank_names{1}).edge_endnode_idx, model.compbanks.(model.tier1_compbank_names{1}).edge_type_filter));
    tstSenseEdges = Edge2Logical(GetEdgeStates(tstDataset.pixels, model.compbanks.(model.tier1_compbank_names{1}).edge_endnode_idx, model.compbanks.(model.tier1_compbank_names{1}).edge_type_filter));
    
    %% render PCA and multidimensional scaling plots
    RenderMDS(outDir, tstDataset, model, tstCode, [], [], [], 'correlation', 'scatter', 'scatter_corr');
    RenderMDS(outDir, tstDataset, model, tstCode, [], [], [], 'euclidean', 'scatter', 'scatter_eucl');
    
    %% render foveated group images
    if ~isempty(trnDataset.img_sz) && any(strcmp(model.compbank_names, 'group')) % if the data is images and we have groups
        try
            for c = 1 : trnDataset.n_classes
                RenderFoveatedGroupImages(outDir, trnDataset.SubsetDatapoints(trnDataset.label_idx == c), model, trnDataset.uniq_classes{c});
            end
        end
    end
   
    %% render TP / FP examples
    predName = {};
    pred = struct();
    try
        [~,pred.onenn] = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trnDataset.label_idx, tstCode.comp_code.(model.output_bank_name)', tstDataset.label_idx, 'knn', knnParams, true);
        predName = [predName,'onenn'];
    catch ex
        warning(ex.message);
    end
    try
        [~,pred.nb]    = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trnDataset.label_idx, tstCode.comp_code.(model.output_bank_name)', tstDataset.label_idx, 'nbfast', nbParams, true);
        predName = [predName,'nb'];
    catch ex
        warning(ex.message);
    end
    try
        [~,pred.net]   = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trnDataset.label_idx, tstCode.comp_code.(model.output_bank_name)', tstDataset.label_idx, 'patternnet', [], true);
        predName = [predName,'net'];
    catch ex
        warning(ex.message);
    end
    try
        [~,pred.svm]   = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trnDataset.label_idx, tstCode.comp_code.(model.output_bank_name)', tstDataset.label_idx, 'svmliblinear', [], true);
        predName = [predName,'svm'];
    catch ex
        warning(ex.message);
    end

    for c = 1 : trnDataset.n_classes
        className = trnDataset.uniq_classes{c};
        for i = 1 : numel(predName)
            idx = find((tstDataset.label_idx(:)' ~= c) & (pred.(predName{i})(:)' == c));
            idx = idx(randperm(numel(idx), min(5, numel(idx)))); % shuffle so we don't wind up with all the zeros and ones (limit to 5 renders per call)
            for j = 1 : numel(idx) % for each datapoint
                RenderDatapoint(outDir, model, tstDataset, tstCode, idx(j), [predName{i},'_tst_fp',className,'-',num2str(j)]);
            end
            
            idx = find((tstDataset.label_idx(:)' == c) & (pred.(predName{i})(:)' == c));
            idx = idx(randperm(numel(idx), min(5, numel(idx)))); % shuffle so we don't wind up with all the zeros and ones (limit to 5 renders per call)
            for j = 1 : numel(idx) % for each datapoint
                RenderDatapoint(outDir, model, tstDataset, tstCode, idx(j), [predName{i},'_tst_tp',className,'-',num2str(j)]);
            end
        end
    end
end