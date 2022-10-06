% performs KNN classification
% if interested in cross-validation, just call ClassifyCrossvalidate(data, label, n_folds, 'knn', ...)
% INPUTS
%   k        - scalar (int-valued numeric) number of neighbors to use
%   data     - n_dims x n_pts (numeric or logical)
%   label    - 1 x n_pts (cell, numeric, or logical) "ground truth" label for each training point
%   distance - (char or function_handle) @(X,Y) function handle or one of 'cityblock', 'chebychev', 'correlation', 'cosine', 'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'minkowski', 'seuclidean' (not sqeuclidean), 'spearman'
% RETURNS
%   predLabel - 1 x n_tstpts (same format as trnLabel)
% see also ml.ClassifyKNN
function predLabel = ClassifyKNNHold1Out(k, data, label, distance)
    validateattributes(k,        {'numeric'},                  {'nonempty','scalar','positive','integer'}, 1);
    validateattributes(data,     {'numeric','logical'},        {'nonempty','2d','nonnan','ncols',numel(label)}, 2);
    validateattributes(label,    {'cell','numeric','logical'}, {'nonempty','vector'}, 3);
    validateattributes(distance, {'char','function_handle'},   {'nonempty'}, 4);

    n_pts = size(data, 2);
    
    idx = NaN(1, n_pts); % index of the training image that best matches each test image
    if k == 1 && strcmpi(distance, 'euclidean')
        if isnumeric(data)
            for i = 1 : n_pts
                difference = sum((data - data(:,i)).^2, 1); % implicit expansion
                difference(i) = Inf; % can't choose self
                [~,idx(i)] = min(difference);
            end
        else % binary values
            for i = 1 : n_pts
                difference = sum(data ~= data(:,i), 1); % implicit expansion
                difference(i) = Inf; % can't choose self
                [~,idx(i)] = min(difference);
            end
        end
    elseif k == 1 && strcmpi(distance, 'cosine')
        l2Norm = vecnorm(double(data), 2, 1); % 1 x n_pts
        if isnumeric(data)
            for i = 1 : n_pts
                similarity = sum(data .* data(:,i), 1) ./ (l2Norm .* l2Norm(i)); % implicit expansion
                similarity(i) = -Inf; % can't choose self
                [~,idx(i)] = max(similarity);
            end
        else % binary values
            for i = 1 : n_pts
                similarity = sum(data & data(:,i), 1) ./ (l2Norm .* l2Norm(i)); % implicit expansion
                similarity(i) = -Inf; % can't choose self
                [~,idx(i)] = max(similarity);
            end
        end
    elseif k == 1 && strcmpi(distance, 'dot') % dot product
        if isnumeric(data)
            for i = 1 : n_pts
                similarity = sum(data .* data(:,i), 1); % implicit expansion
                similarity(i) = -Inf; % can't choose self
                [~,idx(i)] = max(similarity);
            end
        else % binary values
            for i = 1 : n_pts
                similarity = sum(data & data(:,i), 1); % implicit expansion
                similarity(i) = -Inf; % can't choose self
                [~,idx(i)] = max(similarity);
            end
        end
    else
        error('TODO');
    end
    predLabel = label(idx);
end