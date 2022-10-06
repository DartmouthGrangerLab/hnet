% Eli Bowen 8/12/2020
% takes an array of inputs, each of which is considered an independent scalar
% converts to a same-sized array of outputs, where each scalar is represented by a spike based on K-winner-take-all along data dimension dim
% INPUTS
%   data      - any dimensionality (numeric)
%   n_winners - scalar (numeric) number of winners in each competition (i.e. 1 = pure 1WTA)
%   dim       - scalar (numeric) dimension of data along which there's competition
%   min2Win   - OPTIONAL (default = 0) scalar (numeric) the minimum scalar value required for a winner to count (if there aren't n_winners scalars > min2Win, you'll get less than n_winners winners)
%               range is 0-->1, same as range of data
% RETURNS
%   data - logical same size as input data
function data = TransformScalar2SpikeViaKWTA(data, n_winners, dim, min2Win)
    validateattributes(data,      {'numeric'}, {'nonempty'}, 1);
    validateattributes(n_winners, {'numeric'}, {'nonempty','scalar','positive','integer'}, 2);
    validateattributes(dim,       {'numeric'}, {'nonempty','scalar','positive','integer'}, 3);
    assert(min(data(:)) >= 0 && max(data(:)) <= 1, 'input scalar code must be in range 0-->1');
    if ~exist('min2Win', 'var') || isempty(min2Win)
        min2Win = 0;
    else
        assert(numel(min2Win) == 1 && min2Win >= 0 && min2Win <= 1, 'min2Win must be in range 0-->1');
        assert(min2Win < size(data, dim));
    end

    if min2Win > 0
        isAboveMin = (data > min2Win);
    end

    if n_winners == 1
        [~,idx] = max(data, [], dim, 'linear'); % idx contains linear indices into data (but is not a vector itself...)
    else % n_winners > 1
        dataSz = size(data);
        idx = zeros(prod(dataSz((1:numel(dataSz)) ~= dim)), n_winners);
        for i = 1 : n_winners
            [~,idx(:,i)] = max(data, [], dim, 'linear');
            if i < n_winners
                data(idx(:,i)) = -Inf;
            end
        end
    end

    % not using maxk (below) because it doesn't have this fancy linear indexing option
%     [~,idxSub] = maxk(data, n_winners, dim); % idx will have same dimensionality as data, except dimension dim will be size n_winners
%     idx = zeros(size(idxSub));
%     assert(ndims(data) <= 4, 'code not set up to handle data with > 4 dims - please enhance');
%     % there's gotta be a faster way than below...
%     % convert subscripts to linear indices
%     for d4 = 1:size(idx, 4) % extra dimensions will be size 1
%         for d3 = 1:size(idx, 3)
%             for d2 = 1:size(idx, 2)
%                 for d1 = 1:size(idx, 1)
%                     if d1 == dim
%                         idx(d1,d2,d3,d4) = sub2ind(size(data), idx(d1,d2,d3,d4), d2, d3, d4);
%                     elseif d2 == dim
%                         idx(d1,d2,d3,d4) = sub2ind(size(data), d1, idx(d1,d2,d3,d4), d3, d4);
%                     elseif d3 == dim
%                         idx(d1,d2,d3,d4) = sub2ind(size(data), d1, d2, idx(d1,d2,d3,d4), d4);
%                     elseif d4 == dim
%                         idx(d1,d2,d3,d4) = sub2ind(size(data), d1, d2, d3, idx(d1,d2,d3,d4));
%                     end
%                 end
%             end
%         end
%     end

    if isa(data, 'gpuArray')
        data = gpuArray.false(size(data));
    else
        data = false(size(data));
    end
    data(idx(:)') = true;
    if min2Win > 0
        data = data & isAboveMin;
    end
end