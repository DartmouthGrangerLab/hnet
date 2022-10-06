% Eli Bowen 3/2022
% INPUTS
%   idx       - 1 x n (int-valued numeric) NaN or 0 means no category
%   n_classes - OPTIONAL scalar (numeric) number of classes assumed to be indexed by idx DEFAULT = max(idx)
%   is_sparse - OPTIONAL scalar (logical) DEFAULT = false
% RETURNS
%   code - n x n_classes (logical)
function code = OneHot(idx, n_classes, is_sparse)
    validateattributes(idx, {'numeric'}, {'nonnegative'}, 1);
    assert(all(isnan(idx) | mod(idx, 1) == 0)); % must be integer or NaN
    if ~exist('n_classes', 'var') || isempty(n_classes)
        n_classes = max(idx);
    end
    if ~exist('is_sparse', 'var') || isempty(is_sparse)
        is_sparse = false;
    end
    n = numel(idx);

    % way 1
    if is_sparse
        code = logical(sparse(n, n_classes));
    else
        code = false(n, n_classes);
    end
    for i = 1 : n_classes
        code(idx==i,i) = true;
    end

    % way 2
%     code = logical(sparse(idx, 1:n, 1, n_classes, n, n)); % n x n_classes
%     if ~is_sparse
%         code = full(code);
%     end

    % way 3 (requires r2020b or later)
%     code = logical(onehotencode(idx(:)', 1, 'logical', 'ClassNames', 1:n_classes));
%     if is_sparse
%         code = sparse(code);
%     end
end