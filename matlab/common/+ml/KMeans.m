% Eli Bowen
% INPUTS
%   centroids
%   distance   - (char) 'euclidean' | 'squaredeuclidean' | 'cosine' | 'correlation' | 'hamming' | 'jaccard'
%   data       - N x D
%   n_max_iter - scalar (int-valued numeric)
%   do_fuzzy   - OPTIONaL scalar (logical) if true, use fuzzy clustering (default = false) if you don't know what this means, leave false
%   verbose    - OPTIONAL scalar (logical) default = true
% RETURNS
%   clustMemberships - N x 1 integer array with values in the range 1:size(centroids, 1)
%   centroids        - K x D (numeric)
%   iterCentroids    - n_iter x 1 (cell) centroids for each iteration (each cell is the same format as above)
function [clustMemberships,centroids,iterCentroids] = KMeans(centroids, distance, data, n_max_iter, do_fuzzy, verbose)
    validateattributes(centroids, {'numeric','logical'}, {}, 1);
    validateattributes(distance, {'char'}, {}, 2);
    validateattributes(data, {'numeric','logical'}, {}, 3);
    validateattributes(n_max_iter, {'numeric'}, {'nonempty','scalar'}, 4);
    if ~exist('do_fuzzy', 'var') || isempty(do_fuzzy)
        do_fuzzy = false;
    end
    if ~exist('verbose', 'var') || isempty(verbose)
        verbose = true;
    end
    m = 2;

    % remove any all-zero training points
    if any(all(data == 0, 2))
        data(all(data == 0, 2),:) = [];
    end
    
    if strcmp(distance, 'squaredeuclidean')
        distance = 'euclidean';
    end
    is_dist_jaccard = false;
    if strcmp(distance, 'jaccard')
        distance = 'cosine'; % jaccard is cosine with one-hot encoding of the variables
        is_dist_jaccard = true;
        
        [N,D] = size(data); % store now, it'll change below
        temp = cat(1, data, centroids);
        for i = 1 : D
            temp = cat(2, temp, OneHot(temp(:,i)));
        end
        temp(:,1:D) = []; % remove the originals
        data = temp(1:N,:);
        centroids = temp(N+1:end,:);
    end

    if do_fuzzy
        assert(any(strcmp(distance, {'euclidean','cosine','correlation'})), 'currently only euclidean, cosine, correlation, and jaccard kernels are supported.');
        [clustMemberships,centroids,iterCentroids] = ClusterDataFuzzy(data, n_max_iter, centroids, distance, 0.005, verbose, m);
    else
        assert(any(strcmp(distance, {'euclidean','cosine','correlation','hamming'})), 'currently only euclidean, cosine, correlation, hamming, jaccard kernels are supported.');
        [clustMemberships,centroids,iterCentroids] = ClusterData(data, n_max_iter, centroids, distance, 0.005, verbose);
    end

%     javaaddpath(fullfile('/pdata', 'grangerlab', 'ebowen', 'project_videogame', 'MatlabClusterNetwork', 'KMeansMatlabHelper.jar'));
%     import matlabclusternetworkjavahelper.*;
%     if strcmp(distance, 'euclidean')
%         assert(size(data, 1) < 2^31, 'data must have a length less than MAX_INT');
%         helper = matlabclusternetworkjavahelper.KMeansMatlabHelperBlockVectors(K, m_maxIt, numThreads);
%         helper.setChangedThresh(0.005); % .5 percent
%         clustMemberships = helper.ClusterData(data, centroids);
%         centroids = helper.getMeans();
%     elseif strcmp(distance, 'cosine')
%         assert(size(data, 1) < 2^31, 'data must have a length less than MAX_INT');
%         helper = matlabclusternetworkjavahelper.KMeansMatlabHelper(K, m_maxIt, numThreads, kernel);
%         helper.setChangedThresh(0.005); % .5 percent
%         clustMemberships = helper.ClusterData(data, centroids);
%         centroids = double(helper.getMeans());
%         [clustMemberships,centroids] = ClusterData(single(data), m_maxIt, single(centroids), 0, 'euclidean', 0.005);
%         helper = matlabclusternetworkjavahelper.KMeansMatlabHelper(K, m_maxIt, numThreads, 'euclidean');
%         helper.setChangedThresh(0.005); % .5 percent
%         clustMemberships = helper.ClusterData(data, centroids);
%         centroids = double(helper.getMeans());
%     else
%         error('Currently in non-orthogonal mode only euclidean and cosine distance are supported');
%     end
%     centroids = double(centroids);

    % validate
    assert(strcmp(distance, 'correlation') || ~any(isnan(centroids(:))), 'centroids NaN check!');

    % cleanup
    if nargout() > 1 && is_dist_jaccard
        warning('KMeans with jaccard distance will not return centroids or iterCentroids');
        centroids = []; % not valid
        iterCentroids = {}; % not valid
    end
end


% Determine which cluster this data point belongs to, meanwhile update the cluster definitions to incorporate this point
% INPUTS
%   dataPt
%   changeRate - rate at which this new information is incorporated into its centroid. This new data point will have a weight of 1/changeRate
% function [k,centroids] = OnlineCluster (dataPt, centroids, changeRate)
%     k = GetClusterLabel(dataPt);
%     centroids(k,:) = centroids(k,:) .* changeRate;
%     centroids(k,:) = centroids(k,:) + dataPt;
%     centroids(k,:) = centroids(k,:) ./ (changeRate + 1);
% end


% Performs offline clustering of the supplied data
% Can only be called if this clusterer has not already performed clustering
% INPUTS
%   data - dimensionality of #datapts x #dims
function [clustMemberships,centroids,iterCentroids] = ClusterData(data, n_max_iter, centroids, distance, changeFrac, verbose)
    N = size(data, 1);
    K = size(centroids, 1);
    clustMemberships = -ones(N, 1, 'like', data); %initialize to a unique and invalid value
    counts = -ones(K, 1);
    if strcmp(distance, 'cosine')
        % centroids just need to all have the same L2 norm so we can compare cosine distances without norming later (they are not pure means)
        centroids = centroids ./ sqrt(sum(centroids .* centroids, 2)); % L2 norming each row
        % normalize each data point (this is a *choice* so that when computing new means every data point votes equally)
        l2Norms = sqrt(sum(data .* data, 2));
        data = data ./ l2Norms; % L2 norming each row
    elseif strcmp(distance, 'correlation')
        centroids = centroids - mean(centroids, 2);
        centroids = centroids ./ sqrt(sum(centroids .* centroids, 2)); % L2 norming each row
        data = data - mean(data, 2);
        data = data ./ sqrt(sum(data .* data, 2)); % L2 norming each row
        data(isnan(data)) = 0;
    end
    iterCentroids = cell(n_max_iter, 1);

    for i = 1 : n_max_iter
        tic;
        oldLabels = clustMemberships;
        oldCounts = counts;
        [clustMemberships,counts] = ComputeClustMemberships(data, centroids, distance, 0, []);
        n_changed = sum(oldLabels ~= clustMemberships);
        assert(sum(counts==0) >= sum(oldCounts==0));
        time1 = toc();
        if sum(counts>0) < 2
            warning('we only have one cluster!');
            break
        end
        if verbose && sum(counts==0)-sum(oldCounts==0) > 0
            disp(['dropping ',num2str(sum(counts==0)-sum(oldCounts==0)),' cluster(s)']);
        end
        
        tic;
        centroids = ComputeCentroids(centroids, data, K, clustMemberships, counts, distance);
        assert(strcmp(distance, 'correlation') || ~any(isnan(centroids(:))), 'centroids NaN check!');
        time2 = toc();
        
        iterCentroids{i} = centroids;
        if verbose
            disp(['iter ',num2str(i),' changed=',num2str(100 * n_changed / N),'% timing = ',num2str(time1),'s ',num2str(time2),'s ']);
        end
        if n_changed <= changeFrac*N
            if verbose
                disp([num2str(n_changed),' <= ',num2str(changeFrac*N)]);
            end
            break
        end
    end
    if verbose
        disp(['kmeans num changed at end = ',num2str(n_changed),' (',num2str(100 * n_changed / N),'%)']);
    end
end


% Performs offline fuzzy clustering of the supplied data
% https://en.wikipedia.org/wiki/Fuzzy_clustering (euclidean)
% https://www.jstatsoft.org/article/view/v050i10/v50i10.pdf (cosine)
function [clustMemberWeights,centroids,iterCentroids] = ClusterDataFuzzy(data, n_max_iter, centroids, distance, changeFrac, verbose, m)
    N = size(data, 1);
    K = size(centroids, 1);
    clustMemberships = -ones(N, 1, 'like', data);
    clustMemberWeights = zeros(N, K, 'like', data);
    if strcmp(distance, 'cosine')
        % centroids just need to all have the same L2 norm so we can compare cosine distances without norming later (they are not pure means)
        centroids = centroids ./ sqrt(sum(centroids .* centroids, 2)); %L2 norming each row
        % normalize each data point (this is a *choice* so that when computing new means every data point votes equally)
        data = data ./ sqrt(sum(data .* data, 2)); % L2 norming each row
    elseif strcmp(distance, 'correlation')
        centroids = centroids - mean(centroids, 2);
        centroids = centroids ./ sqrt(sum(centroids .* centroids, 2)); % L2 norming each row
        data = data - mean(data, 2);
        data = data ./ sqrt(sum(data .* data, 2)); % L2 norming each row
        data(isnan(data)) = 0;
    end
    iterCentroids = cell(n_max_iter, 1);

    for i = 1 : n_max_iter
        tic;
        oldLabels = clustMemberships;
        clustMemberWeights = ComputeClustMembershipWeights(clustMemberWeights, data, centroids, distance, m, []);
%         n_changed = sum(sum(abs(oldLabels - clustMemberWeights)));
        [~,clustMemberships] = max(clustMemberWeights, [], 2);
        n_changed = sum(oldLabels ~= clustMemberships);
        time1 = toc();
        
        tic;
        centroids = ComputeCentroidsFromWeights(centroids, data, K, clustMemberWeights, distance);
        time2 = toc();
        
        iterCentroids{i} = centroids;
        if verbose
            disp(['iter ',num2str(i),' changed=',num2str(100 * n_changed / N),'% timing = ',num2str(time1),'s ',num2str(time2),'s and ', num2str(sum(sum(clustMemberWeights, 1)<1)),' cluster(s) are dead']);
        end
        if n_changed <= changeFrac*N
            disp([num2str(n_changed),' <= ',num2str(changeFrac*N)]);
            break
        end
    end
    if verbose
        disp(['kmeans num changed at end = ',num2str(n_changed),' (',num2str(100 * n_changed / N),'%)']);
    end
    [~,clustMemberWeights] = max(clustMemberWeights, [], 2);
end


function centroids = ComputeCentroids(centroids, data, K, clustMemberships, counts, distance)
    N = size(data, 1);
    
% %     for i = 1 : N
% %         k = clustMemberships(i);
% %         centroids(k,:) = centroids(k,:) + data(i,:);
% %     end
%     %faster:
%     for k = 1 : K
%         if counts(k) > 0
%             centroids(k,:) = sum(data(clustMemberships==k,:), 1);
%         end
%     end
% %     uniqueClusts = unique(clustMemberships);
% %     tic;
% %     newIdxs = zeros(K, 1);
% %     for i = 1 : numel(uniqueClusts)
% %         newIdxs(uniqueClusts(i)) = i;
% %     end
% %     m_centroids2 = splitapply(@sum, data, newIdxs(clustMemberships));
% %     toc

    % 5x faster than the fastest above method
    % this is some fucking magic from http://www.mathworks.com/matlabcentral/fileexchange/31274-fast-k-means
    if any(strcmp(distance, {'euclidean','cosine','correlation'}))
        [~,~,label] = unique(clustMemberships); % remove empty clusters
        ind = sparse(label, 1:N, 1, K, N, N); % transform label into indicator matrix
        centroids2 = ind * data; % compute centroid of each cluster
        centroids(counts ~= 0,:) = centroids2(1:sum(counts ~= 0),:);
    end

    %% normalize centroid histograms
    if strcmp(distance, 'euclidean')
        for k = 1 : K
            if counts(k) > 0
                centroids(k,:) = centroids(k,:) ./ counts(k);
            end
        end
    elseif strcmp(distance, 'cosine') % centroids just need to all have the same L2 norm so we can compare cosine distances without norming later (they are not pure means)
        for k = 1 : K
            if counts(k) > 0
                norm = sqrt(sum(centroids(k,:) .* centroids(k,:))); % L2 norm
                centroids(k,:) = centroids(k,:) ./ norm;
            end
        end
    elseif strcmp(distance, 'correlation')
        for k = 1 : K
            if counts(k) > 0
                centroids(k,:) = centroids(k,:) - mean(centroids(k,:));
                centroids(k,:) = centroids(k,:) ./ sqrt(sum(centroids(k,:) .* centroids(k,:)));
            end
        end
    elseif strcmp(distance, 'hamming')
        % following matlab's built-in kmeans approach: "each centroid is the component-wise median of points in that cluster"
        for k = 1 : K
            if counts(k) > 0
                centroids(k,:) = median(data(clustMemberships==k,:), 2);
            end
        end
        error('^verify dim');
    else
        error('unexpected distance');
    end
    
    if strcmp(distance, 'correlation')
        centroids(counts == 0,:) = NaN;
    else
        centroids(counts == 0,:) = 0;
    end
end


function centroids = ComputeCentroidsFromWeights(centroids, data, K, clustMemberWeights, distance)    
%     for k = 1:K
%         centroids(k,:) = sum(clustMemberWeights(:,k) .* data, 1);
%     end
    % 100x faster (really)
    centroids = clustMemberWeights' * data;

    %% normalize centroid histograms
    if strcmp(distance, 'euclidean')
        centroids = centroids ./ sum(clustMemberWeights, 1)';
    elseif strcmp(distance, 'cosine') % centroids just need to all have the same L2 norm so we can compare cosine distances without norming later (they are not pure means)
        for k = 1 : K
            norm = sqrt(sum(centroids(k,:) .* centroids(k,:))); % L2 norm
            centroids(k,:) = centroids(k,:) ./ norm;
        end
    elseif strcmp(distance, 'correlation')
        for k = 1 : K
            centroids(k,:) = centroids(k,:) - mean(centroids(k,:));
            centroids(k,:) = centroids(k,:) ./ sqrt(sum(centroids(k,:) .* centroids(k,:)));
        end
        error('TODO: handle degenerate clusters');
    else
        error('unexpected distance');
    end
end


% copied from JavaCommon
% INPUTS
%   data
%   centroids
%   kernel             - (char) 'euclidean' | 'squaredeuclidean' | 'cosine'
%   is_orthogonal      - scalar (logical)
%   n_input_categories - OPTIONAL unless is_orthogonal == true
function [clustMemberships,counts] = ComputeClustMemberships(data, centroids, kernel, is_orthogonal, n_input_categories)
    K = size(centroids, 1);
    
    if is_orthogonal
        error('hasnt been tested in forever');
%         if parType == 2
%             if strcmp(kernel, 'squaredeuclidean')
%                 kernel = 'euclidean';
%             end
% %             javaaddpath(fullfile('/pdata', 'grangerlab', 'ebowen', 'project_videogame', 'MatlabClusterNetwork', 'MatlabClusterNetworkJavaHelper.jar'));
%             import matlabclusternetworkjavahelper.*;
%             helper = matlabclusternetworkjavahelper.ClustMembershipsComputerOrthogonal(size(centroids, 1), kernel, DetermineNumJavaComputeCores());
%             clustMemberships = helper.ComputeClustMemberships(int32(data-1), centroids, n_input_categories);
%         else
%             error('invalid parType');
%         end
    else
        if strcmp(kernel, 'squaredeuclidean')
            kernel = 'euclidean';
        end
        clustMemberships = ComputeClustMembershipsHelper(data, centroids, kernel);
    end

    if nargout > 1
        counts = CountNumericOccurrences(clustMemberships, 1:K);
    end
end


function clustMemberships = ComputeClustMembershipsHelper(data, centroids, distance)
%     N = size(data, 1);
%     K = size(centroids, 1);

    if strcmp(distance, 'euclidean')
%         for i = 1 : K
%             precompCentroids(i) = sum(centroids(i,:) .* centroids(i,:)) * 0.5;
%         end
        precompCentroids = sum(centroids .* centroids, 2)' .* 0.5;
        deadCentroids = all(centroids==0, 2);
        precompCentroids(deadCentroids) = Inf; %so they'll never win
    end
    
%     clustMemberships = ones(N, 1, 'like', data);
%     bestDist = zeros(N, 1, 'like', data);
%     if strcmp(distance, 'euclidean')
%         for k = 1:K
%             dists4Clust = precompCentroids(k) - (data * centroids(k,:)');
%             if k == 1
%                 bestDist = dists4Clust;
%             else
%                 winners = dists4Clust < bestDist; % greater than (>) because dot product is a *similarity* measure not distance
%                 bestDist(winners) = dists4Clust(winners);
%                 clustMemberships(winners) = k;
%             end
%         end
%     elseif strcmp(distance, 'cosine')
%         % assumes all centroids are of same L2 norm length (should be)
%         for k = 1:K
%             dists4Clust = data * centroids(k,:)';
%             if k == 1
%                 bestDist = dists4Clust;
%             else
%                 winners = dists4Clust > bestDist; % greater than (>) because dot product is a *similarity* measure not distance
%                 bestDist(winners) = dists4Clust(winners);
%                 clustMemberships(winners) = k;
%             end
%         end
%         [~,clustMemberships] = min(pdist2(data, centroids, 'cosine'), [], 2); % D times more memory intensive, but at least 3x faster
%     end
    % D times more memory intensive, 15x faster
    if strcmp(distance, 'euclidean')
        [~,clustMemberships] = min(precompCentroids - (data * centroids'), [], 2); % D times more memory intensive, 15x faster
    elseif strcmp(distance, 'cosine') || strcmp(distance, 'correlation')
        [~,clustMemberships] = max(data * centroids', [], 2); % D times more memory intensive, 15x faster
    elseif strcmp(distance, 'hamming')
        [~,clustMemberships] = max(sum(data == centroids, 2));
        error('^verify');
    end
    % the jointstills way
    % gamma is sorted like *similarity*
%     if strcmp(distance, 'euclidean')
%         Gamma = -precompCentroids + (data * centroids');
%     elseif strcmp(distance, 'cosine')
%         Gamma = data * centroids';
%     end
%     Gamma = exp(Gamma - max(Gamma, [], 2)); %max for each datapt
%     Gamma = Gamma ./ sum(Gamma, 2);
%     error('^verify we''re taking the sum of responses to a single datapoint');
end


% copied from JavaCommon
% INPUTS
%   data
%   centroids - K x D
%   distance  - (char) 'euclidean' | 'squaredeuclidean' | 'cosine'
%   m
function clustMemberWeights = ComputeClustMembershipWeights(clustMemberWeights, data, centroids, distance, m)
    %% compute distance
    if strcmp(distance, 'euclidean')
        % yes, the below is squared euclidean dist as planned
        clustMemberWeights = bsxfun(@plus, sum(data.*data, 2), sum(centroids.*centroids, 2)') - 2*(data*centroids'); % 4x as fast as pdist2(data, centroids, 'euclidean')
    elseif strcmp(distance, 'cosine')
        clustMemberWeights = 1 - (data * centroids');
    elseif strcmp(distance, 'correlation')
        clustMemberWeights = (2 - (data * centroids')) ./ 2;
    end
    clustMemberWeights = max(clustMemberWeights, 0); % can be within floating point error of 0, but negative
    clustMemberWeights = clustMemberWeights .^ (1/(m-1));

    %% compute membership weights
%     pdists = clustMemberWeights;
%     for j = 1 : K
%         for i = 1 : N
%             clustMemberWeights(i,j) = sum(pdists(i,j) ./ pdists(i,:));
%         end
%     end
%     clustMemberWeights = 1 ./ clustMemberWeights;
    % 10x faster than above
%     for j = 1:K
%         clustMemberWeights(:,j) = sum(pdists(:,j) ./ pdists, 2);
%     end
%     clustMemberWeights = 1 ./ clustMemberWeights;
    % 10x even faster
    clustMemberWeights = 1 ./ (clustMemberWeights .* sum(1 ./ clustMemberWeights, 2));

    %% handle situations where datapoint is identical to centroid (as recommended by https://www.jstatsoft.org/article/view/v050i10/v50i10.pdf)
    numZeros = sum(isnan(clustMemberWeights), 2);
    clustMemberWeights(numZeros~=0,:) = isnan(clustMemberWeights(numZeros~=0,:));
    clustMemberWeights(numZeros>1,:) = clustMemberWeights(numZeros>1,:) ./ numZeros(numZeros>1);

    %% cleanup
    temp = sum(clustMemberWeights, 2);
    assert(all(temp < 1.0001) && all(temp > 0.9999));

    clustMemberWeights = clustMemberWeights .^ m; % so we don't have to do this multiple times later
end
