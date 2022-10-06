% Eli Bowen 8/12/2020
% takes an array of inputs, each of which is considered an independent scalar
% converts to a larger array of outputs, where each scalar is represented by a spatial code (Gaussian receptive fields arranged along the line)
% INPUTS
%   data            - any dimensionality (numeric)
%   n_spatial_stops - scalar (numeric) - e.g. 5, number of stops to make along the scalar continuum (more means more specificity, more output dims)
%   meta            - OPTIONAL (struct) metadata, each nonscalar field must be the same dimensionality as data
% RETURNS
%   data        - size(squeeze(data)) x n_spatial_stops (numeric; same class as input)
%   spatialStop - size(squeeze(data)) x n_spatial_stops (numeric)
%   meta        - scalar (struct)
function [data,spatialStop,meta] = TransformScalar2SpatialScalar(data, n_spatial_stops, meta)
    validateattributes(data, {'numeric'}, {'nonempty'}, 1);
    validateattributes(n_spatial_stops, {'numeric'}, {'nonempty','scalar','positive','integer'}, 2);
    assert(min(data(:)) >= 0 && max(data(:)) <= 1, 'input scalar code must be in range 0-->1');
    data = squeeze(data);
    n_used_dims = sum(size(data) > 1);
    if n_used_dims == 1
        data = data(:); % place vector along dim 1
    end

    % another way to do it, with quantiles:
%     quantiles = zeros(numel(0:1/n_spatial_stops:0.9999), size(data, 2));
%     for i = 1 : size(data, 2)
%         tempData = data(:,i);
%         quantiles(:,i) = quantile(tempData(tempData~=0), 0:1/n_spatial_stops:0.9999, 1);
%     end
%     quantiles = sort(quantiles, 1, 'ascend'); % stupid matlab bug - quantile doesn't always produce sorted results (only almost always)
% 
%     for i = 1 : size(data, 2)
%         data(:,i) = discretize(data(:,i), [-Inf;quantiles(:,i);Inf]);
%     end

    d = cell(1, n_spatial_stops);
    for i = 1 : n_spatial_stops
        d{i} = normpdf(data, i/n_spatial_stops, i/(2*n_spatial_stops)) ./ normpdf(i/n_spatial_stops, i/n_spatial_stops, i/(2*n_spatial_stops));
        % note there is no detector for scalar=0, by preference
    end
    data = cat(n_used_dims + 1, d{:});

    % spatial stop number
    if nargout() > 1
        spatialStop = zeros(size(data));
        if n_used_dims == 1
            [~,spatialStop(:)] = ind2sub(size(data), 1:numel(data));
        elseif n_used_dims == 2
            [~,~,spatialStop(:)] = ind2sub(size(data), 1:numel(data));
        elseif n_used_dims == 3
            [~,~,~,spatialStop(:)] = ind2sub(size(data), 1:numel(data));
        elseif n_used_dims == 4
            [~,~,~,~,spatialStop(:)] = ind2sub(size(data), 1:numel(data));
        elseif n_used_dims == 5
            [~,~,~,~,~,spatialStop(:)] = ind2sub(size(data), 1:numel(data));
        else
            error('havent bothered supporting > 5D inputs');
        end
    end

    if nargout() > 2
        if ~exist('meta', 'var') || isempty(meta)
            meta = struct();
        end

        r = [ones(1, n_used_dims),n_spatial_stops];
        fn = fieldnames(meta);
        for i = 1 : numel(fn)
            if numel(meta.(fn{i})) > 1
                temp = squeeze(meta.(fn{i}));
                if n_used_dims == 1
                    temp = temp(:);
                end
                meta.(fn{i}) = repmat(temp, r);
            end
        end
    end
end