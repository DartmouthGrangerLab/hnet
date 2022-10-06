% Eli Bowen
% randomly subsets each category (except the rarest one) so that they all have equal N
% INPUTS:
%   labelIdx - 1 x n_datapts (numeric index)
%   n        - OPTIONAL scalar (int-valued numeric) number of items per class (if not specified, will use as many as possible)
% RETURNS:
%   selectedIdx - 1 x n_returned_datapts (int-valued double)
function selectedIdx = EqualizeN(labelIdx, n)
    validateattributes(labelIdx, {'numeric'}, {'integer'}, 1);

    [uniqLabel,~,labelIdx] = unique(labelIdx);

    if ~exist('n', 'var') || isempty(n) || isnan(n)
        n = min(CountNumericOccurrences(labelIdx, 1:numel(uniqLabel)));
    else
        validateattributes(n, 'numeric', {'scalar','nonnegative','integer'});
        assert(n <= min(CountNumericOccurrences(labelIdx, 1:numel(uniqLabel))));
    end

    % form separate arrays for each category, subset to be of equal length, then recombine
    selectedIdx = zeros(n, numel(uniqLabel));
    for i = 1 : numel(uniqLabel)
        classIdx = find(labelIdx == i);
        selectedIdx(:,i) = classIdx(randperm(numel(classIdx), n));
    end

    selectedIdx = sort(selectedIdx(:)'); % vectorize, be stable
end