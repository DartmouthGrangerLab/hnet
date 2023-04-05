% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   model
%   trnpixels     - n_px x n_trn (logical)
%   trnlabelidx   - n_trn x 1 (numeric index)
%   tstpixels     - n_px x n_tst (logical)
%   tstlabelidx   - n_tst x 1 (numeric index)
%   trndat        - scalar (Dataset)
%   tstdat        - scalar (Dataset)
%   trn_comp_code - ? x ? (?)
%   tst_comp_code - ? x ? (?)
%   modelName
%   append - (char) text to append to output file names
function [] = PrintPerformance(model, trnpixels, trnlabelidx, tstpixels, tstlabelidx, trn_comp_code, tst_comp_code, modelName, append)
    arguments
        model(1,1) Model, trnpixels(:,:), trnlabelidx(:,1), tstpixels(:,:), tstlabelidx(:,1), trn_comp_code(:,:), tst_comp_code(:,:), modelName(1,1) string, append(1,1) string
    end
    senseDidx = model.compbanks.(model.tier1_compbank_names{1}).edge_endnode_idx;

    edgeTypeFilter = model.compbanks.(model.tier1_compbank_names{1}).edge_type_filter;
    
    trnSense = Edge2Logical(GetEdgeStates(trnpixels, senseDidx, edgeTypeFilter));
    tstSense = Edge2Logical(GetEdgeStates(tstpixels, senseDidx, edgeTypeFilter));
    
    classifier = 'knn';
    params = struct(k=1, distance='dot');
    trntrnInAcc = ml.ClassifyHold1Out(trnpixels', trnlabelidx, classifier, params, true);
    [trntstInAcc,trntstInPred] = ml.Classify(trnpixels', trnlabelidx, tstpixels', tstlabelidx, classifier, params, true);

    trntrnEdgeAcc = ml.ClassifyHold1Out(trnSense', trnlabelidx, classifier, params, true);
    trntstEdgeAcc = ml.Classify(trnSense', trnlabelidx, tstSense', tstlabelidx, classifier, params, true);
    
    if islogical(trn_comp_code) || isa(trn_comp_code, "uint8")
        params = struct(k=1, distance='dot');
    else % feat codes are probably energies; ~poisson distributed
        params = struct(k=1, distance='cosine');
    end
    trntrnAcc = NaN;
    trntstAcc = NaN;
    p_us_vs_in = NaN;
    try
        trntrnAcc = ml.ClassifyHold1Out(trn_comp_code', trnlabelidx, classifier, params, true);
        [trntstAcc,trntstPred] = ml.Classify(trn_comp_code', trnlabelidx, tst_comp_code', tstlabelidx, classifier, params, true);
        [~,p_us_vs_in] = testcholdout(trntstPred, trntstInPred, tstlabelidx, 'Alternative', 'unequal', 'Test', 'midp'); % 2-sided mid-p-value McNemar Test
    end
    
    s = struct(analysis=char(modelName),...
        hnet_tst_acc=round(trntstAcc, 4),...
        edge_tst_acc=round(trntstEdgeAcc, 4),...
        pixel_tst_acc=round(trntstInAcc, 4),...
        hnet_trn_acc=round(trntrnAcc, 4),...
        edge_trn_acc=round(trntrnEdgeAcc, 4),...
        pixel_trn_acc=round(trntrnInAcc, 4),...
        timestamp=char(datetime("now")),...
        p_us_vs_in=p_us_vs_in);
    io.InjectRowIntoTableFile(Config.OUT_DIR, char("acc_1nn_" + append + ".csv"), 'analysis', s);
    
    
    classifier = 'nbfast';
    params = struct(distribution='bern');
    trntrnInAcc = ml.Classify(trnpixels', trnlabelidx, trnpixels', trnlabelidx, classifier, params, true);
    [trntstInAcc,trntstInPred] = ml.Classify(trnpixels', trnlabelidx, tstpixels', tstlabelidx, classifier, params, true);
    
    trntrnEdgeAcc = ml.Classify(trnSense', trnlabelidx, trnSense', trnlabelidx, classifier, params, true);
    trntstEdgeAcc = ml.Classify(trnSense', trnlabelidx, tstSense', tstlabelidx, classifier, params, true);
    
    if islogical(trn_comp_code) || isa(trn_comp_code, "uint8")
        params = struct(distribution='bern');
    else % feat codes are probably energies; ~poisson distributed
        params = struct(distribution='gauss');
    end
    trntrnAcc  = NaN;
    trntstAcc  = NaN;
    p_us_vs_in = NaN;
    try
        trntrnAcc = ml.Classify(trn_comp_code', trnlabelidx, trn_comp_code', trnlabelidx, classifier, params, true);
        [trntstAcc,trntstPred] = ml.Classify(trn_comp_code', trnlabelidx, tst_comp_code', tstlabelidx, classifier, params, true);
        [~,p_us_vs_in] = testcholdout(trntstPred, trntstInPred, tstlabelidx, 'Alternative', 'unequal', 'Test', 'midp'); % 2-sided mid-p-value McNemar Test
    end

    s = struct(analysis=char(modelName),...
        hnet_tst_acc=round(trntstAcc, 4),...
        edge_tst_acc=round(trntstEdgeAcc, 4),...
        pixel_tst_acc=round(trntstInAcc, 4),...
        hnet_trn_acc=round(trntrnAcc, 4),...
        edge_trn_acc=round(trntrnEdgeAcc, 4),...
        pixel_trn_acc=round(trntrnInAcc, 4),...
        timestamp=char(datetime("now")),...
        p_us_vs_in=p_us_vs_in);
    io.InjectRowIntoTableFile(Config.OUT_DIR, char("acc_nb_" + append + ".csv"), 'analysis', s);
    

    classifier = 'svmliblinear';
    params = struct(regularization_lvl=1);
    trntrnInAcc = ml.Classify(trnpixels', trnlabelidx, trnpixels', trnlabelidx, classifier, params, true);
    [trntstInAcc,trntstInPred] = ml.Classify(trnpixels', trnlabelidx, tstpixels', tstlabelidx, classifier, params, true);

    trntrnEdgeAcc = ml.Classify(trnSense', trnlabelidx, trnSense', trnlabelidx, classifier, params, true);
    trntstEdgeAcc = ml.Classify(trnSense', trnlabelidx, tstSense', tstlabelidx, classifier, params, true);

    trntrnAcc  = NaN;
    trntstAcc  = NaN;
    p_us_vs_in = NaN;
    try
        trntrnAcc = ml.Classify(trn_comp_code', trnlabelidx, trn_comp_code', trnlabelidx, classifier, params, true);
        [trntstAcc,trntstPred] = ml.Classify(trn_comp_code', trnlabelidx, tst_comp_code', tstlabelidx, classifier, params, true);
        [~,p_us_vs_in] = testcholdout(trntstPred, trntstInPred, tstlabelidx, 'Alternative', 'unequal', 'Test', 'midp'); % 2-sided mid-p-value McNemar Test
    end
    
    s = struct(analysis=char(modelName),...
        hnet_tst_acc=round(trntstAcc, 4),...
        edge_tst_acc=round(trntstEdgeAcc, 4),...
        pixel_tst_acc=round(trntstInAcc, 4),...
        hnet_trn_acc=round(trntrnAcc, 4),...
        edge_trn_acc=round(trntrnEdgeAcc, 4),...
        pixel_trn_acc=round(trntrnInAcc, 4),...
        timestamp=char(datetime("now")),...
        p_us_vs_in=p_us_vs_in);
    io.InjectRowIntoTableFile(Config.OUT_DIR, char("acc_svm_" + append + ".csv"), 'analysis', s);
    
    
    classifier = 'patternnet';
    trntrnInAcc   = zeros(10, 1);
    trntstInAcc   = zeros(10, 1);
    trntrnEdgeAcc = zeros(10, 1);
    trntstEdgeAcc = zeros(10, 1);
    trntrnAcc     = zeros(10, 1);
    trntstAcc     = zeros(10, 1);
    p_us_vs_in    = zeros(10, 1);
    for i = 1 : 10 % varies strongly based on random initialization
        try
            trntrnInAcc(i) = ml.Classify(trnDataset.pixels', trnlabelidx, trnDataset.pixels', trnlabelidx, classifier, [], true);
            [trntstInAcc(i),trntstInPred] = ml.Classify(trnDataset.pixels', trnlabelidx, tstDataset.pixels', tstlabelidx, classifier, [], true);
            trntrnEdgeAcc(i) = ml.Classify(trnSense', trnlabelidx, trnSense', trnlabelidx, classifier, [], true);
            trntstEdgeAcc(i) = ml.Classify(trnSense', trnlabelidx, tstSense', tstlabelidx, classifier, [], true);
        catch % in case deep learning toolbox not present
            trntrnInAcc(i)   = NaN;
            trntstInAcc(i)   = NaN;
            trntrnEdgeAcc(i) = NaN;
            trntstEdgeAcc(i) = NaN;
        end
        
        try
            trntrnAcc(i) = ml.Classify(trn_comp_code', trnlabelidx, trn_comp_code', trnlabelidx, classifier, [], true);
            [trntstAcc(i),trntstPred] = ml.Classify(trn_comp_code', trnlabelidx, tst_comp_code', tstlabelidx, classifier, [], true);
            p_us_vs_in(i) = NaN;
            [~,p_us_vs_in(i)] = testcholdout(trntstPred, trntstInPred, tstlabelidx, 'Alternative', 'unequal', 'Test', 'midp'); % 2-sided mid-p-value McNemar Test
        catch
            trntrnAcc(i)  = NaN;
            trntstAcc(i)  = NaN;
            p_us_vs_in(i) = NaN;
        end
    end
    trntrnInAcc     = mean(trntrnInAcc);
    trntstInAcc     = mean(trntstInAcc);
    trntrnEdgeAcc   = mean(trntrnEdgeAcc);
    trntstEdgeAcc   = mean(trntstEdgeAcc);
    trntrnAcc       = mean(trntrnAcc);
    trntstAcc       = mean(trntstAcc);
    p_us_vs_in_mean = mean(p_us_vs_in);

    
    s = struct(analysis=char(modelName),...
        hnet_tst_acc=round(trntstAcc, 4),...
        edge_tst_acc=round(trntstEdgeAcc, 4),...
        pixel_tst_acc=round(trntstInAcc, 4),...
        hnet_trn_acc=round(trntrnAcc, 4),...
        edge_trn_acc=round(trntrnEdgeAcc, 4),...
        pixel_trn_acc=round(trntrnInAcc, 4),...
        timestamp=char(datetime("now")),...
        p_us_vs_in_mean=p_us_vs_in_mean);
    io.InjectRowIntoTableFile(Config.OUT_DIR, char("acc_net_" + append + ".csv"), 'analysis', s);
end