% performs KNN classification
% INPUTS
%   k        - scalar (int-valued numeric) number of neighbors to use
%   trnData  - n_dims x n_trnpts (numeric or logical)
%   tstData  - n_dims x n_tstpts (numeric or logical)
%   trnLabel - 1 x n_trnpts (cell, numeric, or logical) "ground truth" label for each training point
%   distance - (char or function_handle) @(X,Y) function handle or one of 'cityblock', 'chebychev', 'correlation', 'cosine', 'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'minkowski', 'seuclidean' (not sqeuclidean), 'spearman'
% RETURNS
%   predLabel    - 1 x n_tstpts (same format as trnLabel)
%   predStrength - 1 x n_tstpts (double ranged 0-->1) a measure of how strongly KNN believes in the prediction in predLabel (SLOW TO COMPUTE) (ONLY returned for euclidean or cosine distance)
% see also ml.Classify, ml.ClassifyCrossvalidate, ml.ClassifyHold1Out
function [predLabel,predStrength] = ClassifyKNN(K, trnData, tstData, trnLabel, distance)
    validateattributes(K,        {'numeric'},                  {'nonempty','scalar','positive','integer'}, 1);
    validateattributes(trnData,  {'numeric','logical'},        {'nonempty','2d','nonnan','nrows',size(tstData, 1),'ncols',numel(trnLabel)}, 2);
    validateattributes(tstData,  {'numeric','logical'},        {'nonempty','2d','nonnan','nrows',size(trnData, 1)}, 3);
    validateattributes(trnLabel, {'cell','numeric','logical'}, {'nonempty','vector'}, 4);
    validateattributes(distance, {'char','function_handle'},   {'nonempty'}, 5);
    assert((isnumeric(trnData) && isnumeric(tstData)) || (islogical(trnData) && islogical(tstData))); % inputs must be same datatype

    n_tst_pts = size(tstData, 2);
    
    if K == 1
        predLabel = ml.OneNN(trnData, tstData, trnLabel, distance);
    else
        model = fitcknn(trnData, trnLabel, 'ClassNames', uniqueLabels, 'NumNeighbors', k, 'Distance', distance);
        predLabel = predict(model, tstData);
        error('TODO: validate (never tested)');
    end
    
    predStrength = [];
    if nargout() > 1 && (strcmpi(distance, 'euclidean') || strcmpi(distance, 'cosine') || strcmpi(distance, 'dot'))
        diameter = min(n_tst_pts, K * 10); % in units of number of points
        w = 1 ./ (1:1/K:1+(diameter-1)/K); % weight for each point, after they're ranked from closest to furthest
        w = w ./ sum(w); % normalize so max is 1, min is 0
        predStrength = NaN(size(predLabel));
        if strcmpi(distance, 'euclidean')
            for i = 1 : n_tst_pts
                if isnumeric(trnData)
                    [~,idx] = sort(sum((trnData - tstData(:,i)).^2, 1), 'ascend'); % implicit expansion
                else % binary values
                    [~,idx] = sort(sum(trnData ~= tstData(:,i), 1), 'ascend'); % implicit expansion
                end
                temp = (predLabel(i) == trnLabel(idx(1:diameter)));
                predStrength(i) = sum(temp(:)' .* w(:)');
            end
        elseif strcmpi(distance, 'cosine')
            trnL2Norm = vecnorm(double(trnData), 2, 1); % 1 x n_trn_pts
            tstL2Norm = vecnorm(double(tstData), 2, 1); % 1 x n_tst_pts
            for i = 1 : n_tst_pts
                if isnumeric(trnData)
                    [~,idx] = sort(sum(trnData .* tstData(:,i), 1) ./ (trnL2Norm.*tstL2Norm(i)), 'descend'); % implicit expansion
                else % binary values
                    [~,idx] = sort(sum(trnData & tstData(:,i), 1) ./ (trnL2Norm.*tstL2Norm(i)), 'descend'); % implicit expansion
                end
                temp = (predLabel(i) == trnLabel(idx(1:diameter)));
                predStrength(i) = sum(temp(:)' .* w(:)');
            end
        elseif strcmpi(distance, 'dot') % dot product
            for i = 1 : n_tst_pts
                if isnumeric(trnData)
                    [~,idx] = sort(sum(trnData .* tstData(:,i), 1), 'descend'); % implicit expansion
                else % binary values
                    [~,idx] = sort(sum(trnData & tstData(:,i), 1), 'descend'); % implicit expansion
                end
                temp = (predLabel(i) == trnLabel(idx(1:diameter)));
                predStrength(i) = sum(temp(:)' .* w(:)');
            end
        end
    end
end