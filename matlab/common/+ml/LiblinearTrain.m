% Eli Bowen 9/12/2018
% wrapper designed to simplify and remove the chance of errors when calling liblinear's train() function
% designed to work with liblinear-multicore version 2.43-2
% switched from version 2.20 (still available as a zip) on 9/29/2021
% on windows, I only have the compiler working via visual studio
% note liblinear uses its own RNG, which can't be controlled
% USAGE
%   model = TrainLiblinear(0, labelNum, data, true, 1);
%   [predLabel,acc,score,mse,sqcorr] = LiblinearPredict(model, labelNum, data);
% INPUTS
%   solverType - char or numeric, one of:
%       For multi-class classification:
%       0 / 'logreg' -- L2-regularized logistic regression (primal)
%       1 -- L2-regularized L2-loss support vector classification (dual)
%       2 / 'svm' -- L2-regularized L2-loss support vector classification (primal)
%       3 -- L2-regularized L1-loss support vector classification (dual)
%       4 (NOT SUPPORTED YET) -- support vector classification by Crammer and Singer
%       5 -- L1-regularized L2-loss support vector classification
%       6 -- L1-regularized logistic regression
%       7 (MAYBE SUPPORTED) -- L2-regularized logistic regression (dual)
%       For regression:
%       11 -- L2-regularized L2-loss support vector regression (primal)
%       12 (NOT SUPPORTED YET) -- L2-regularized L2-loss support vector regression (dual)
%       13 (NOT SUPPORTED YET) -- L2-regularized L1-loss support vector regression (dual)
%   for outlier detection
%       21 (NOT SUPPORTED YET) -- one-class support vector machine (dual)
%   label             - 1 x N (int-valued numeric) vector of category IDs for the data
%   data              - N x D (numeric or logical)
%   doAdjust4UnequalN - scalar (logical) if true, categories will be weighted so that rare categories get the same importance as common categories
%   regularizationLvl - scalar (numeric) how heavily to weight regularization (liblinear's default was = 1). Set to eps to almost disable regularization, but it'll suck. Set to 'optimize' to have liblinear find the highest performing regularizationLevel.
% RETURNS
%   model - a struct as described in the 4 liblinear README files. The LAST feature is the bias/intercept term, which you may wish to remove. This struct will change in form depending on whether you passed 2 categories or more than 2
%       .???
%       .w
%       .bias
%       .norm_min
%       .norm_max
function model = LiblinearTrain(solverType, label, data, doAdjust4UnequalN, regularizationLvl)
    if ischar(solverType)
        if strcmp(solverType, 'logreg')
            solverType = 0;
        elseif strcmp(solverType, 'svm')
            solverType = 2;
        else
            error('unexpected solverType');
        end
    end
    validateattributes(solverType,        {'numeric'},           {'nonempty','scalar','nonnegative','integer'}, 1);
    validateattributes(label,             {'numeric'},           {'nonempty','vector','positive','integer'}, 2);
    validateattributes(data,              {'numeric','logical'}, {'nonempty','2d','nonnan','nrows',numel(label)}, 3);
    validateattributes(doAdjust4UnequalN, {'logical'},           {'nonempty','scalar'}, 4);
    validateattributes(regularizationLvl, {'numeric','char'},    {'nonempty'}, 5);
    if any(solverType == 4:10)
        warning('liblinear-multicore options 4 through 10 were never tested');
    end
    assert((isnumeric(regularizationLvl) && isscalar(regularizationLvl)) || strcmp(regularizationLvl, 'optimize'));
    assert(~isa(data, 'gpuArray'), 'liblinear doesn''t have gpu support');

    n_cores = DetermineNumJavaComputeCores();
    if ~any(solverType == [0,1,2,3,5,6,11]) % only ones implemented by the version of the parallel library we use
        n_cores = 1;
    end
    if ispc() % cant get liblinear_multicore to compile for windows properly
        n_cores = 1;
    end
    
    label = label(:); % required orientation...
    N = numel(label);
    [uniqueLabel,~,labelNum] = unique(label);
    assert(all(labelNum == label)); % only required because otherwise the results of the model will be misinterpreted by the calling function
    counts = CountNumericOccurrences(labelNum, 1:numel(uniqueLabel));
    if islogical(data)
        normMin = 0;
        normMax = 1;
    else
        normMin = min(data, [], 1);
        data = data - normMin; % must pre-scale for runtime and accuracy
        normMax = max(data, [], 1);
        data = data ./ normMax; % must pre-scale for runtime and accuracy
    end
    data = double(data);
    data = sparse([data,ones(N, 1)]); % sparse required by the alg, performance is often poor without a col of ones at the end

    weightStr = ''; % using weightStr is probably more fair, but it's unclear
    if doAdjust4UnequalN
        for i = 1 : numel(uniqueLabel)
            if numel(uniqueLabel) == 2
                weightStr = [weightStr,' -w',num2str(i),' ',num2str(N / counts(i))];
            else
                if counts(i) > 0
                    weightStr = [weightStr,' -w',num2str(i),' ',num2str(1 / (counts(i)*(1/(N-counts(i)))))];
                end
            end
        end
    end
    solverTypeStr = num2str(solverType);

    if strcmp(regularizationLvl, 'optimize')
        assert(any(solverType == [0,2,11]), 'currently, optimized regularizationLvl only supported by liblinear for solverType 0, 2, 11');
        if n_cores == 1
            regularizationLvl = train_liblinear(label, data, ['-q -s ',solverTypeStr,' -C -v 5',weightStr]); % appending a bias/intercept term
        else
            regularizationLvl = train_liblinear_multicore(label, data, ['-q -s ',solverTypeStr,' -C -v 5 -m ',num2str(n_cores),weightStr]); % appending a bias/intercept term
        end
        regularizationLvl = regularizationLvl(1); % returns [best_C,best_p,best_score]
    end
    regularizationLvl = num2str(regularizationLvl);

    if n_cores == 1
        model = train_liblinear(label, data, ['-q -s ',solverTypeStr,' -c ',regularizationLvl,weightStr]); % appending a bias/intercept term
    else
        model = train_liblinear_multicore(label, data, ['-q -s ',solverTypeStr,' -c ',regularizationLvl,' -m ',num2str(n_cores),weightStr]); % appending a bias/intercept term
    end

    assert(~isempty(model)); % if model is empty, liblinear crashed
    assert(model.bias < 0); % I think this is always -1 (aka "ignore me") unless we specify the bias beforehand, which we wouldn't normally do

    model.norm_min = normMin;
    model.norm_max = normMax;
end