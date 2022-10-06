% multinomial naive bayes, translated from scikit-learn
% https://github.com/scikit-learn/scikit-learn/blob/c3dd4051dda40b28877877a2405806c77502c18d/sklearn/naive_bayes.py#L508
% INPUTS
%   X        - d x n (numeric) data matrix
%   labelIdx - 1 x n (int-valued numeric) label (1~k)
% RETURNS
%   model
function model = nbMulti(X, labelIdx)
    validateattributes(X,        {'numeric'}, {'nonempty'}, 1);
    validateattributes(labelIdx, {'numeric'}, {'nonempty','vector'}, 2);
    assert(size(X, 2) == numel(labelIdx));
    assert(~any(X(:) < 0));
    labelIdx = labelIdx(:)';
    
    n = size(X, 2);
    k = max(labelIdx); % n_classes

    E = full(sparse(labelIdx, 1:n, 1, k, n, n)); % create a 1-hot label code (n x k)
    nk = sum(E, 2);             % k x 1 (int-valued numeric) num datapoints per class
    w = log(nk) - log(sum(nk)); % k x 1 (numeric) class log prior
    dia = diag(1 ./ nk, 0);     % k x k (numeric) diagonal matrix (with values of 1./nk along the diagonal)
    feature_count = X * (E' * dia); % * dia ADDED BY ELI NOT IN scikit CODE!!! (seems more right I guess?)
%     feature_count = X * E'; % count raw events from data before updating the class log prior and feature log probas
    alpha = 1; % additive (Laplace/Lidstone) smoothing parameter
%     alpha = 1e-10; % smallest reccomended value
    mu = log(feature_count + alpha) - log(sum(feature_count + alpha, 2)); % compute feature log probabilities
    
    model = struct();
    model.dist = 'multinomial';
    model.w = w;   % k x 1
    model.mu = mu; % d x k
end


% dot product that handle the sparse matrix case correctly
% INPUTS
%   a : {ndarray, sparse matrix}
%   b : {ndarray, sparse matrix}
% RETURNS
%   dot_product : {ndarray, sparse matrix}
function ret = safe_sparse_dot(a, b)
    if a.ndim > 2 || b.ndim > 2
        ret = np.dot(a, b);
    else
        ret = a @ b;
    end
end