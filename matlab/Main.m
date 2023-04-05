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
%   Main("metacred",  "ucicreditgerman", "tier1.memorize-->tier1.extractcorr.icacropsome.100.50.unsupsplit-->meta.extractcorr.kmeans.10.50.unsupsplit");
%   Main("groupedimg", "mnistpy.128", "connectedpart.memorize-->connectedpart.extractconnec.25-->connectedpart.transl.2");
%   Main("clevrpos1", "clevrpossimple", "tier1.memorize");
function [] = Main(modelName, frontendSpec, trnSpec)
    assert(endsWith(Config.MyDir(), "/matlab") || endsWith(Config.MyDir(), "\matlab"), "Main() expects to be run from the hnet/matlab/ directory");
    addpath(genpath(Config.MyDir())); % add sub-folders in case they weren't added
    
    cfg = struct(model_name=char(modelName), frontend_spec=char(frontendSpec), trn_spec=char(trnSpec));
    outDir = fullfile(Config.OUT_DIR, cfg.model_name + "_" + cfg.frontend_spec + "_" + strrep(cfg.trn_spec, "-->", "-"));
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    disp("=== " + outDir + " ===");
    
    layout = Layout(modelName);
    
    %% load dataset
    SetRNG(1000);
    trndat = Dataset(cfg.frontend_spec, "trn");
    tstdat = Dataset(cfg.frontend_spec, "tst");
    
    % print dataset info to text files
    temp = cat(2, trndat.pixel_metadata.name, num2cell(sum(trndat.pixels, 2)));
    writecell(cat(1, {'node_name','num nonzero pixels'}, temp), fullfile(Config.OUT_DIR, "node_name_" + cfg.frontend_spec + ".csv"));
    if isfield(trndat.other_metadata, "category_info")
        temp = cat(2, fieldnames(trndat.other_metadata.category_info), struct2cell(trndat.other_metadata.category_info));
        writecell(cat(1, {'field','value'}, temp), fullfile(Config.OUT_DIR, "category_info_" + cfg.frontend_spec + ".csv"));
    end
    if isfield(trndat.other_metadata, "t")
        writetable(trndat.other_metadata.t, fullfile(Config.OUT_DIR, cfg.frontend_spec + "_trn.csv"));
        writetable(tstdat.other_metadata.t, fullfile(Config.OUT_DIR, cfg.frontend_spec + "_tst.csv"));
    end
    if isfield(trndat.other_metadata, "t_bin")
        writetable(trndat.other_metadata.t_bin, fullfile(Config.OUT_DIR, cfg.frontend_spec + "_bin_trn.csv"));
        writetable(tstdat.other_metadata.t_bin, fullfile(Config.OUT_DIR, cfg.frontend_spec + "_bin_tst.csv"));
    end
    t = table(trndat.pixel_metadata.name(:));
    for i = 1 : trndat.n_classes
        t = addvars(t, round(sum(trndat.pixels(:,trndat.label_idx == i), 2) ./ trndat.n_pts, 4), NewVariableNames=['frac_occurs_in_pts_of_class_',trndat.uniq_classes{i}]);
    end
    writetable(t, fullfile(Config.OUT_DIR, cfg.frontend_spec + "_var_cls_cooccur_frequency.csv"));
    
    % render example datapoints
    for c = 1 : trndat.n_classes
        idx = find(trndat.label_idx == c);
        for i = 1 : min(numel(idx), 5) % for each image in this class
            RenderDatapointNoEdges(fullfile(Config.OUT_DIR, "samplestim"), trndat, idx(i), trndat.uniq_classes{c}+"_"+num2str(i))
        end
    end
    
    %% train
    if Config.DO_CACHE
        model = CachedCompute(@Train, cfg, layout, trndat); % with caching
    else
        model = Train(cfg, layout, trndat); % without caching
    end
    
    SetRNG(cfg); % reset RNG after training
    
    %% encode
    trnCode = struct();
    tstCode = struct();

    [trnCode.comp_code,trnCode.premerge_idx] = Encode(model, trndat.pixels);
    [tstCode.comp_code,tstCode.premerge_idx] = Encode(model, tstdat.pixels);

    if modelName == "clevrpos1" || modelName == "clevrpos2"
        for i = 1 : 100
            RenderCLEVRPosImg(outDir, trndat, model, i);
        end
        return
    elseif modelName == "clevr"
        return
    end

    trnCode.hist = struct();
    tstCode.hist = struct();
    trnCode.comp_best_img = struct();
    tstCode.comp_best_img = struct();
    for i = 1 : model.n_compbanks
        name = model.compbank_names{i};

        trnCode.hist.(name) = ClassHistogram(encode.OneHot(trndat.label_idx, trndat.n_classes), trnCode.comp_code.(name));
        tstCode.hist.(name) = ClassHistogram(encode.OneHot(tstdat.label_idx, tstdat.n_classes), tstCode.comp_code.(name));

        [~,trnCode.comp_best_img.(name)] = max(trnCode.comp_code.(name), [], 2);
        [~,tstCode.comp_best_img.(name)] = max(tstCode.comp_code.(name), [], 2);
    end

    %% print accuracy and related stats
    PrintPerformance(model, trndat.pixels, trndat.label_idx, tstdat.pixels, tstdat.label_idx, trnCode.comp_code.(model.output_bank_name), tstCode.comp_code.(model.output_bank_name), cfg.model_name + "_" + cfg.trn_spec, cfg.frontend_spec);

    %% render the edges involved in each component
    for i = 1 : model.n_compbanks
        bank = model.compbank_names{i};
        
        RenderAll(outDir, model, tstdat, tstCode, bank, true, true);
        RenderAll(outDir, model, tstdat, tstCode, bank, true, false);
        RenderAll(outDir, model, tstdat, tstCode, bank, false, true);

        % get metadata and pass to PrintEdgeRelations
        node_info = model.compbanks.(bank).g.node_metadata.name;
        if strcmp(bank, model.tier1_compbank_names{1}) && isfield(tstdat.other_metadata, "category_info")
            for j = 1 : tstdat.n_nodes
                if isfield(tstdat.other_metadata.category_info, node_info{j})
                    node_info{j} = tstdat.other_metadata.category_info.(node_info{j});
                end
            end
        end

        for j = 1 : min(100, model.compbanks.(bank).n_cmp) % for each component in this bank
            PrintEdgeRelations(outDir, bank+"_cmp"+int2str(j)+".txt", model.compbanks.(bank).edge_states(:,j), model.compbanks.(bank).edge_endnode_idx, node_info, bank, model.tier1_compbank_names{1});
        end
    end

    %% explain each tst datapoint
    if modelName == "metacred"
        RenderTstToExplain(outDir, model, tstdat, tstCode, model.output_bank_name, true);
        RenderTstToExplain(outDir, model, tstdat, tstCode, model.output_bank_name, false);
    end
    
    %% render component best matches
    for i = 1 : numel(model.tier1_compbank_names)
        % can only call for tier 1 and tier 2 component banks
        RenderComponentBestMatches(outDir, model, tstdat, tstCode, model.tier1_compbank_names{i});
        [~,downstreamBankNames] = outedges(model.g, model.tier1_compbank_names{i});
        for j = 1 : numel(downstreamBankNames)
            if ~strcmp(downstreamBankNames{j}, "out")
                RenderComponentBestMatches(outDir, model, tstdat, tstCode, downstreamBankNames{j});
            end
        end
    end
    
    %% render class histograms
    for i = 1 : model.n_compbanks
        RenderHist(outDir, trndat.uniq_classes, trnCode, model.compbank_names{i}, "trn");
        RenderHist(outDir, trndat.uniq_classes, tstCode, model.compbank_names{i}, "tst");
    end

    %% render discriminability vs sharedness of each component's response to the dataset
    if ~isfield(model.compbanks, "meta") && ~isfield(model.compbanks, "group") && trndat.n_classes > 2 % if we only have one tier
        RenderDiscrimVsSharednessVsFrequency(outDir, model, trndat, trnCode, "trn");
        RenderDiscrimVsSharednessVsFrequency(outDir, model, tstdat, tstCode, "tst");
    end
    
    %% render the groups
    if isfield(model.compbanks, "group")
        for i = 1 : min(10, model.compbanks.group.n_cmp) % for each group
            RenderGroup(outDir, model, tstdat, tstCode, i);
        end
    end
    
    %% render PCA and multidimensional scaling plots
    RenderMDS(outDir, tstdat, model, tstCode, [], [], [], "correlation", "scatter", "scatter_corr");
    RenderMDS(outDir, tstdat, model, tstCode, [], [], [], "euclidean", "scatter", "scatter_eucl");
    
    %% render foveated group images
    if ~isempty(trndat.img_sz) && any(strcmp(model.compbank_names, "group")) % if the data is images and we have groups
        try
            for c = 1 : trndat.n_classes
                RenderFoveatedGroupImages(outDir, trndat.SubsetDatapoints(trndat.label_idx == c), model, trndat.uniq_classes{c});
            end
        end
    end

    %% precompute some stats for below figures
    if islogical(trnCode.comp_code.(model.output_bank_name)) || isa(trnCode.comp_code.(model.output_bank_name), "uint8")
        knnParams = struct(k=1, distance='dot');
        nbParams = struct(distribution='bern');
    else % feat codes are probably energies; ~poisson distributed
        knnParams = struct(k=1, distance='cosine');
        nbParams = struct(distribution='gauss');
    end
    trnSenseEdges = Edge2Logical(GetEdgeStates(trndat.pixels, model.compbanks.(model.tier1_compbank_names{1}).edge_endnode_idx, model.compbanks.(model.tier1_compbank_names{1}).edge_type_filter));
    tstSenseEdges = Edge2Logical(GetEdgeStates(tstdat.pixels, model.compbanks.(model.tier1_compbank_names{1}).edge_endnode_idx, model.compbanks.(model.tier1_compbank_names{1}).edge_type_filter));
    try
        predName = {'onenn','nb','net','svm'};
        pred = struct();
        [~,pred.onenn] = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trndat.label_idx, tstCode.comp_code.(model.output_bank_name)', tstdat.label_idx, 'knn', knnParams, true);
        [~,pred.nb]    = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trndat.label_idx, tstCode.comp_code.(model.output_bank_name)', tstdat.label_idx, 'nbfast', nbParams, true);
        [~,pred.net]   = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trndat.label_idx, tstCode.comp_code.(model.output_bank_name)', tstdat.label_idx, 'patternnet', [], true);
        [~,pred.svm]   = ml.Classify(trnCode.comp_code.(model.output_bank_name)', trndat.label_idx, tstCode.comp_code.(model.output_bank_name)', tstdat.label_idx, 'svmliblinear', [], true);
    catch ex
        warning(ex.message);
        disp("terminating figure code early due to above issue");
        return
    end

    %% render TP / FP examples
    for c = 1 : trndat.n_classes
        className = trndat.uniq_classes{c};
        for i = 1 : numel(predName)
            idx = find((tstdat.label_idx(:)' ~= c) & (pred.(predName{i})(:)' == c));
            idx = idx(randperm(numel(idx), min(5, numel(idx)))); % shuffle so we don't wind up with all the zeros and ones (limit to 5 renders per call)
            for j = 1 : numel(idx) % for each datapoint
                RenderDatapoint(outDir, model, tstdat, tstCode, idx(j), predName{i}+"_tst_fp"+className+"-"+num2str(j));
            end
            
            idx = find((tstdat.label_idx(:)' == c) & (pred.(predName{i})(:)' == c));
            idx = idx(randperm(numel(idx), min(5, numel(idx)))); % shuffle so we don't wind up with all the zeros and ones (limit to 5 renders per call)
            for j = 1 : numel(idx) % for each datapoint
                RenderDatapoint(outDir, model, tstdat, tstCode, idx(j), predName{i}+"_tst_tp"+className+"-"+num2str(j));
            end
        end
    end
end