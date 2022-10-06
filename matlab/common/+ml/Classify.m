% Eli Bowen 10/1/2021
% INPUTS
%   trnData        - n_trnpts x n_dims (numeric or logical)
%   trnLabel       - 1 x n_trnpts (int-valued numeric or cellstr)
%   tstData        - n_tstpts x n_dims (numeric or logical)
%   tstLabel       - 1 x n_tstpts (int-valued numeric or cellstr)
%   classifierType - (char) 'lda' | 'svm' | 'svmjava' | 'svmliblinear' | 'logreg' | 'logregliblinear' | 'knn' | 'nb' | 'nbfast' | 'perceptron' | 'patternnet' | 'decisiontree'
%   params         - OPTIONAL (struct)
%       .cost                  - misclassification cost, a KxK matrix where first dim is true label, second dim is predicted label (default: ones(K) - eye(K))
%       .k                     - for knn (numeric) DEFAULT = 1
%       .distance              - for knn (char) 'euclidean' | 'correlation' | 'cosine' | 'hamming' | ...
%       .distribution          - for nbfast (char) 'bern' | 'gauss' | 'multinomial'
%       .hidden_sz             - for patternnet (numeric) see patternnet(); DEFAULT = [10] (one layer with 10 nodes)
%       .train_func            - for patternnet (char) see patternnet(); DEFAULT = 'trainscg'
%       .perform_func          - for patternnet (char) see patternnet(); DEFAULT = 'crossentropy'
%       .n_variables_to_sample - for decisiontree (numeric or 'all') see templateTree(); DEFAULT = 'all'
%       .method                - for decisiontree (char) see fitcensemble(); DEFAULT = 'Bag'
%       .n_learning_cycles     - for decisiontree (numeric) see fitcensemble(); DEFAULT = 100
%   verbose - OPTIONAL scalar (logical) should we print text? (default=false)
% RETURNS
%   acc - scalar (double ranged 0 --> 1) accuracy (mean across folds)
%   predLabel
%   score - n_tstpts x n_classes (numeric) 'score(i,j) represents the confidence that data point i is of class j'
function [acc,predLabel,score] = Classify(trnData, trnLabel, tstData, tstLabel, classifierType, params, verbose)
    validateattributes(trnData,        {'numeric','logical'}, {'nonempty','2d','nonnan','nrows',numel(trnLabel),'ncols',size(tstData, 2)}, 1);
    validateattributes(trnLabel,       {'numeric'},           {'nonempty','vector'}, 2);
    validateattributes(tstData,        {'numeric','logical'}, {'nonempty','2d','nonnan','nrows',numel(tstLabel),'ncols',size(trnData, 2)}, 3);
    validateattributes(tstLabel,       {'numeric'},           {'nonempty','vector'}, 4);
    validateattributes(classifierType, {'char'},              {'nonempty'}, 5);
    if ~exist('params', 'var') || isempty(params)
        params = struct();
    end
    if ~isfield(params, 'distribution')
        if islogical(trnData) || all(trnData(:) == 0 | trnData(:) == 1)
            params.distribution = 'bern';
        else
            params.distribution = 'gauss';
        end
    end
    if ~isfield(params, 'regularization_lvl');    params.regularization_lvl = 'optimize'; end
    if ~isfield(params, 'k');                     params.k = 1; end
    if ~isfield(params, 'distance');              params.distance = 'euclidean'; end
    if ~isfield(params, 'hidden_sz');             params.hidden_sz = 10; end
    if ~isfield(params, 'train_func');            params.train_func = 'trainscg'; end
    if ~isfield(params, 'perform_func');          params.perform_func = 'crossentropy'; end
    if ~isfield(params, 'n_variables_to_sample'); params.n_variables_to_sample = 'all'; end
    if ~isfield(params, 'method');                params.method = 'Bag'; end
    if ~isfield(params, 'n_learning_cycles');     params.n_learning_cycles = '100'; end
    if ~exist('verbose', 'var') || isempty(verbose)
        verbose = false;
    end
    assert((islogical(trnData) && islogical(tstData)) || (~islogical(trnData) && ~islogical(tstData)));

    %% clean up, standardize provided labels
    % some algs require labels to be the integers 1:n_classes
    trnLabel = trnLabel(:); % input label can be either 1 x N or N x 1, below code requires consistency
    tstLabel = tstLabel(:); % input label can be either 1 x N or N x 1, below code requires consistency
    [uniqLabel,~,trnLabelIdx] = unique(trnLabel); % convert labels into index into uniqLabels
    n_classes = numel(uniqLabel);
    
    tstLabelIdx = tstLabel;
    [r,c] = find(uniqLabel(:) == tstLabel(:)'); % untested may be faster
    tstLabelIdx(c) = r;
    % above is faster than below, same result
%     for i = 1 : numel(tstLabel)
%         tstLabelIdx(i) = find(uniqLabel == tstLabel(i));
%     end

    n_trn = numel(trnLabelIdx);
    n_tst = numel(tstLabelIdx);
    
    %% remove dimensions of zero variance (or learning algs will crash)
    variances = var(trnData, 0, 1);
    if any(variances == 0)
        assert(~all(variances == 0));
        trnData(:,variances == 0) = [];
        tstData(:,variances == 0) = [];
    end
    
    %% prepare variables
    cost = [];
    if isfield(params, 'cost') && ~isempty(params.cost)
        cost = params.cost;
    end
    do_parallel = true;
    
    %% print info
    if verbose
        disp([classifierType,'...']);
        disp(uniqLabel(:)');
        if any(variances == 0)
            disp(['removed ',num2str(sum(variances==0)),' dims with zero variance']);
        end
        t = tic();
    end

    %% classify
    acc = [];
    score = []; % in case we don't set it below
    if strcmp(classifierType, 'nb') % --- naive bayes via matlab ---
        if islogical(trnData)
            error('ues classifierType=nbfast with logical data; classifierType=nb doesnt support bernoulli distributions');
        end
        model = fitcnb(trnData, trnLabelIdx, 'ClassNames', uniqLabel, 'Cost', cost);
        acc = 1 - loss(model, tstData, tstLabelIdx, 'LossFun', 'classiferror'); % loss = percent incorrect for each fold on testing data
        if nargout > 1 % for efficiency, only get predLabel and scores if necessary
            [predLabel,~,score] = predict(model, tstData, 'Options', statset('UseParallel', do_parallel));
        end
        warning('nb untested');
    elseif strcmp(classifierType, 'nbfast') % --- naive bayes via faster 3rd party lib ---
        if islogical(trnData)
            trnData = double(trnData); % can't be single
            tstData = double(tstData); % can't be single
        end
        if strcmp(params.distribution, 'bern') % bernoulli, for boolean/binary data
            model = ml.nbBern(trnData', trnLabelIdx(:)');
        elseif strcmp(params.distribution, 'gauss') % gaussian-distributed data
            model = ml.nbGauss(trnData', trnLabelIdx(:)');
        elseif strcmp(params.distribution, 'multinomial') % for count data
            model = ml.nbMulti(trnData', trnLabelIdx(:)');
        else
            error('unexpected distribution');
        end
        predLabel = ml.nbPred(model, tstData')';
    elseif strcmp(classifierType, 'lda') % --- lda via matlab ---
        if islogical(trnData)
            trnData = double(trnData);
            tstData = double(tstData);
        end
        if n_classes == 2 % 2-class lda
            model = fitcdiscr(trnData, trnLabelIdx, 'ClassNames', uniqLabel, 'Cost', cost, 'discrimType', 'pseudoLinear');
        else % multiclass lda
            model = fitcecoc(trnData, trnLabelIdx, 'ClassNames', uniqLabel, 'Cost', cost, 'Learners', templateDiscriminant('discrimType', 'pseudoLinear'), 'Options', statset('UseParallel', do_parallel));
        end
        acc = 1 - loss(model, tstData, tstLabelIdx, 'LossFun', 'classiferror'); % loss = percent incorrect for each fold on testing data
        if nargout > 1 % for efficiency, only get predLabel and scores if necessary
            [predLabel,~,score] = predict(model, tstData, 'Options', statset('UseParallel', do_parallel));
        end
        error('lda untested');
    elseif strcmp(classifierType, 'svm') % --- svm via matlab ---
        if islogical(trnData)
            trnData = double(trnData);
            tstData = double(tstData);
        end
        if n_classes == 2 % 2-class svm
            model = fitcsvm(trnData, trnLabelIdx, 'ClassNames', uniqLabel, 'Cost', cost, 'KernelFunction', 'linear', 'Standardize', true);
        else % multiclass svm
            model = fitcecoc(trnData, trnLabelIdx, 'ClassNames', uniqLabel, 'Cost', cost, 'Learners', templateSVM('Standardize', 1, 'KernelFunction', 'linear'), 'Options', statset('UseParallel', do_parallel));
        end
        acc = 1 - loss(model, tstData, tstLabelIdx, 'LossFun', 'classiferror'); % loss = percent incorrect for each fold on testing data
        if nargout > 1 % for efficiency, only get predLabel and scores if necessary
            [predLabel,~,score] = predict(model, tstData); % no parallel option for ficsvm
        end
    elseif strcmp(classifierType, 'svmjava') % --- svm via java ---
        error('not yet implemented');
    elseif strcmp(classifierType, 'svmliblinear') % --- svm via liblinear-multicore ---
        doAdjust4UnequalN = true;
        model = ml.LiblinearTrain('svm', trnLabelIdx, trnData, doAdjust4UnequalN, params.regularization_lvl);
        [predLabel,acc,score,~,~] = ml.LiblinearPredict(model, tstLabelIdx, tstData);
    elseif strcmp(classifierType, 'logreg') % --- logistic regression via matlab ---
        error('not yet implemented');
    elseif strcmp(classifierType, 'logregliblinear') % --- logistic regression via liblinear-multicore
        doAdjust4UnequalN = true;
        model = ml.LiblinearTrain('logreg', trnLabelIdx, trnData, doAdjust4UnequalN, params.regularization_lvl);
        [predLabel,acc,score,~,~] = ml.LiblinearPredict(model, tstLabelIdx, tstData);
    elseif strcmp(classifierType, 'knn') % --- knn ---
        if nargout() > 2 % for efficiency, only calc scores if needed
            [predLabel,score] = ml.ClassifyKNN(params.k, trnData', tstData', trnLabelIdx, params.distance);
        else
            predLabel         = ml.ClassifyKNN(params.k, trnData', tstData', trnLabelIdx, params.distance);
        end
        % for knn, score is the "strength" of the classification
    elseif strcmp(classifierType, 'perceptron') % --- a single perceptron ---
        assert(n_classes == 2, 'classifier type perceptron only runs on 2-class problems');
        model = perceptron();
        model.trainParam.showWindow = 0; % disable gui
        % 'useParallel' and 'useGPU': 'no' is default, can set to 'yes'
        model = train(model, trnData', encode.OneHot(trnLabelIdx, n_classes)', 'CheckpointDelay', 0); % data must be d x n, labels must be one-hot n_classes x n
        predLabel = model(tstData');
        [~,predLabel] = max(predLabel, [], 1); % convert from one-hot to idx
    elseif strcmp(classifierType, 'patternnet') % --- patternnet ---
        model = patternnet(params.hidden_sz, params.train_func, params.perform_func); % default trainFcn = 'trainscg', performFcn = 'crossentropy'
        model.trainParam.showWindow = 0; % disable gui
        model = train(model, trnData', encode.OneHot(trnLabelIdx, n_classes)', 'CheckpointDelay', 0); % data must be d x n, labels must be one-hot
        predLabel = sim(model, tstData');
        [~,predLabel] = max(predLabel, [], 1); % convert from one-hot to idx
        % can also use feedforwardnet for more fine-grained control (patternnet is a kind of feedforwardnet)
    elseif strcmp(classifierType, 'decisiontree') % --- ensemble of decision trees ---
        t = templateTree('NumVariablesToSample', params.n_variables_to_sample, 'PredictorSelection', 'allsplits', 'Reproducible', true);
        model = fitcensemble(trnData, trnLabelIdx, 'Method', params.method, 'NumLearningCycles', params.n_learning_cycles, 'Learners', t); % can also do 'Options', statset(.)
        predLabel = predict(model, tstData);
    else
        error('unexpected classifierType');
    end

    %% finalize
    if isempty(acc)
        acc = sum(predLabel(:) == tstLabelIdx(:)) / n_tst; % for most models
    end
    acc = gather(acc);
    if nargout > 1 % for efficiency, only get predLabel and scores if necessary
        predLabel = uniqLabel(predLabel); % convert back from idx into uniqLabel to labels as they were provided in the input
    end
    if nargout > 2
        score = gather(score);
    end

    Toc(t, verbose, [num2str(n_classes),'-class ',num2str(size(trnData, 2)),'-dim ',classifierType,', n_trn = ',num2str(n_trn),', n_tst = ',num2str(n_tst),', acc = ',num2str(acc, 4)]);
end