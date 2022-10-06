% Eli Bowen 1/16/17
% NOTE: counts has the same number of elements (and same order), as unique(x)
% omits nans
% INPUTS
%   x       - ? x ? (numeric, usually int-valued)
%   uniqArr - OPTIONAL, default = unique(x)
%   dim     - OPTIONAL, default = 1, only used when x is a matrix
% RETURNS
%   counts - number of entries for each unique number, same dimensionality as uniqArr
function counts = CountNumericOccurrences(x, uniqArr, dim)
    validateattributes(x, {'numeric'}, {}, 1);
    if ~exist('uniqArr', 'var') || isempty(uniqArr)
        uniqArr = unique(x(:));
        uniqArr(isnan(uniqArr)) = [];
    else
        validateattributes(uniqArr, {'numeric'}, {'nonempty','vector','nonnan'});
        assert(numel(uniqArr) == numel(unique(uniqArr)) && issorted(uniqArr, 'ascend'));
    end
    n_uniq = numel(uniqArr);
    
    if isempty(x)
        counts = zeros(size(uniqArr), 'like', uniqArr);
        return
    elseif ~isvector(x)
        if ~exist('dim', 'var') || isempty(dim) || dim == 1
            counts = zeros(size(x, 2), n_uniq);
            for i = 1 : size(x, 2)
                counts(:,i) = CountNumericOccurrences(x(:,i), uniqArr); % recurse
            end
        else
            assert(dim == 2, 'if provided, dim must be 1 or 2');
            counts = zeros(size(x, 1), n_uniq);
            for i = 1 : size(x, 1)
                counts(i,:) = CountNumericOccurrences(x(i,:), uniqArr); % recurse
            end
        end
        return
    end
    
    if any(isnan(x))
        x(isnan(x)) = []; % omit nans (for efficiency, don't change x if not needed)
    end

    if n_uniq == max(uniqArr) && all(uniqArr(:)' == 1:n_uniq)
        % below is suuuper fast, but to use it with uniqArr ~= 1:n_unique_nums we need to run unique with nargout=3 which is slow
        counts = accumarray(x(:), ones(numel(x), 1), [n_uniq,1])';
    elseif numel(x) * n_uniq * 64 / 8 / 1024 / 1024 / 1024 < DetermineComputerRAMSize()
        counts = sum(x(:)'==uniqArr(:), 2)'; % implicit expansion, 2x as fast as loops (but briefly creates large matrices sometimes)
    else
        % this will be very slow whenever called
        counts = zeros(1, n_uniq);
        for i = 1 : n_uniq
            counts(i) = sum(x==uniqArr(i));
        end
    end
    % another way is below (2x speed of the slowest method, but it needs major debugging)
%     [~,idxa,idxc] = unique(x);
%     [countsTemp,~,idxTemp] = histcounts(idxc, numel(idxa));

    if size(uniqArr, 2) == 1
        counts = counts'; % output vector in same row/col orientation as input
    end
end