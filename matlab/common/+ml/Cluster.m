% Eli Bowen 11/30/2019
% copied and enhanced from LSRExperiments() (Eli Bowen, 7/2/2019)
% USAGE
%   see ml.ClustInit()
% INPUTS
%   model    - (struct) result of ClustInit()
%   data     - N x D (numeric or logical)
%   do_fuzzy - OPTIONAL scalar (logical)
%   n_iter   - OPTIONAL scalar (numeric)
% RETURNS
%   model - (struct) model with modified clusters
%   data  - N x K (numeric) similarity between each cluster and each datapoint
%   idx   - N x 1 (numeric index) index of the cluster closest to each datapoint
% see also ml.ClustInit
function [model,data,idx] = Cluster(model, data, do_fuzzy, n_iter)
    validateattributes(model, {'struct'}, {'nonempty'}, 1);
    if ~isfloat(data)
        data = double(data);
    end
    if ~exist('do_fuzzy', 'var') || isempty(do_fuzzy)
        do_fuzzy = false;
    end
    if do_fuzzy
        assert(strcmp(model.clusterer, 'kmeans'));
    end

    if strcmp(model.distance, 'cosine') || strcmp(model.distance, 'jaccard')
        % fix zeroish-length vectors
        Xnorm = realsqrt(sum(data.^2, 2));
        tIdxs = find(Xnorm <= eps(max(Xnorm)));
        data = [data,zeros(size(data, 1), 1)];
        data(tIdxs,end) = 1;
    end

    if strcmp(model.clusterer, 'kmeans')
        [idx,model.mu,~] = ml.KMeans(model.mu, model.distance, data, n_iter, do_fuzzy, false);
    elseif strcmp(model.clusterer, 'gmm')
        [idx,model] = ml.GMM(data, model.k, n_iter, model);
    elseif startsWith(model.clusterer, 'hierarchical')
        linkage = strrep(model.clusterer, 'hierarchical', ''); % linkage - (char) 'average' | 'centroid' | 'complete' | 'median' | 'single' | 'ward' | 'weighted'
        idx = clusterdata(data, 'MaxClust', model.k, 'Distance', model.distance, 'Linkage', linkage);
    elseif startsWith(model.clusterer, 'spectralkmeans')
        assert(size(data, 1) == size(data, 2)); % data must be a square adjacency matrix
        assert(strcmp(model.distance, 'euclidean') || strcmp(model.distance, 'sqeuclidean'));
        type = str2double(model.clusterer(end));
        [idx,model.mu,model.L,model.U] = ml.SpectralClustering(data, model.k, type, model.init, n_iter); % mu is K x N, L is the same dimensionality as data (N x N), U is ___
    else
        error('unknown clusterer');
    end

    counts = CountNumericOccurrences(idx, 1:model.k);

    if any(counts == 0)
        warning(['removing ',num2str(sum(counts==0)),' (of ',num2str(model.k),') clusters for having 0 members']);

        if strcmp(model.clusterer, 'kmeans')
            model.mu(counts==0,:) = [];
        elseif strcmp(model.clusterer, 'gmm')
            if strcmp(model.modelType, 'matlabbuiltin')
                error('TODO');
            elseif strcmp(model.modelType, 'em_matlabcentral')
                model.mu(:,counts==0)      = [];
                model.sigma(:,:,counts==0) = [];
                model.w(counts==0)         = [];
            else
                error('invalid model.modelType');
            end
        elseif startsWith(model.clusterer, 'hierarchical')
            % nothing to do
        elseif startsWith(model.clusterer, 'spectralkmeans')
            model.mu(counts==0,:) = [];
        else
            error('unknown clusterer');
        end
    end

    %% calculate responses
    if nargout() > 1
        if startsWith(model.clusterer, 'hierarchical')
            [~,idx] = min(data, [], 2); % idx must be re-computed
            data = double(OneHot(idx, model.k));
        else
            data = ml.ClustResponse(model, data); % data are not *similarities*
%             figure;imagesc(normalize(data, 2, 'range'));colorbar
            if nargout() > 2
                [~,idx] = max(data, [], 2); % idx must be re-computed
            end
        end
    end
end