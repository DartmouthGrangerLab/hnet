% performs simple 1NN classification
% INPUTS
%   trnData  - n_dims x n_trnpts (numeric or logical)
%   tstData  - n_dims x n_tstpts (numeric or logical)
%   trnLabel - n_trnpts x 1 (cell, numeric, or logical) "ground truth" label for each training point
%   distance - (char) 'euclidean' | 'cosine' | 'dot' | 'correlation' | 'spearman' | 'cityblock' | 'hamming' | 'jaccard'
% RETURNS
%   predLabel - n_tstpts x 1 (same format as trnLabel)
% see also ClassifyKNN
function predLabel = OneNN(trnData, tstData, trnLabel, distance)
    validateattributes(trnData,  {'numeric','logical'},        {'nonempty','2d','nonnan','nrows',size(tstData, 1),'ncols',numel(trnLabel)}, 1);
    validateattributes(tstData,  {'numeric','logical'},        {'nonempty','2d','nonnan','nrows',size(trnData, 1)}, 2);
    validateattributes(trnLabel, {'cell','numeric','logical'}, {'nonempty','vector'}, 3);
    validateattributes(distance, {'char','function_handle'},   {'nonempty'}, 4);
    assert((isnumeric(trnData) && isnumeric(tstData)) || (islogical(trnData) && islogical(tstData))); % inputs must be same datatype

    n_tst_pts = size(tstData, 2);
    
    idx = NaN(n_tst_pts, 1); % index of the training image that best matches each test image
    
    if strcmpi(distance, 'euclidean')
        if isnumeric(trnData)
            for i = 1 : n_tst_pts
                [~,idx(i)] = min(sum((trnData - tstData(:,i)).^2, 1)); % implicit expansion
            end
        else % bool values
            for i = 1 : n_tst_pts
                [~,idx(i)] = min(sum(trnData ~= tstData(:,i), 1)); % implicit expansion
            end
        end
    elseif strcmpi(distance, 'cosine')
        trnL2Norm = vecnorm(double(trnData), 2, 1); % 1 x n_trn_pts
        tstL2Norm = vecnorm(double(tstData), 2, 1); % 1 x n_tst_pts
        if isnumeric(trnData)
            for i = 1 : n_tst_pts
                [~,idx(i)] = max(sum(trnData .* tstData(:,i), 1) ./ (trnL2Norm.*tstL2Norm(i))); % implicit expansion
            end
        else % bool values
            for i = 1 : n_tst_pts
                [~,idx(i)] = max(sum(trnData & tstData(:,i), 1) ./ (trnL2Norm.*tstL2Norm(i))); % implicit expansion
            end
        end
    elseif strcmpi(distance, 'dot') % dot product
        if isnumeric(trnData)
            for i = 1 : n_tst_pts
                [~,idx(i)] = max(sum(trnData .* tstData(:,i), 1)); % implicit expansion
            end
        else % bool values
            for i = 1 : n_tst_pts
                [~,idx(i)] = max(sum(trnData & tstData(:,i), 1)); % implicit expansion
            end
        end
    elseif strcmpi(distance, 'correlation') % pearson
        error('TODO');
    elseif strcmpi(distance, 'spearman')
        error('TODO');
    elseif strcmpi(distance, 'cityblock') % L1 norm
        error('TODO');
    elseif strcmpi(distance, 'hamming')
        error('TODO');
    elseif strcmpi(distance, 'jaccard')
        error('TODO');
    else
        error('unexpected distance');
    end
    
    predLabel = trnLabel(idx);
end