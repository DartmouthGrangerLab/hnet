% Eli Bowen 11/30/2019
% USAGE
%   model = ml.ClustInit(trnData, k, 'clusterer', 'distance', init);
%   [model,responsesToTrnData] = ml.Cluster(model, trnData, do_fuzzy, maxIter);
%   responsesToTstData = ml.ClustResponse(model, tstData);
% INPUTS
%   clusterer - (char) 'kmeans' | 'gmm' | 'hierarchical<linkage>' | 'spectralkmeans<1|2|3>'
%   k         - scalar (int-valued numeric) number of clusters
%   distance  - (char) any valid distance measure e.g. 'euclidean'
%   init      - OPTIONAL (numeric, struct, char) if numeric, a cluster idx for each datapoint. if struct, a pre-existing model. if char, one of 'random', 'furthestfirst', 'kmeans++'
%   data      - OPTIONAL N x D (numeric)
% RETURNS
%   model - struct
%       .clusterer
%       .distance
%       .mu
%       [other fields]
% see also ml.Cluster
function model = ClustInit(clusterer, K, distance, init, data)
    validateattributes(clusterer, {'char'}, {'nonempty'}, 1);
    validateattributes(K, {'numeric'}, {'nonempty','scalar','integer'}, 2);
    validateattributes(distance, {'char'}, {'nonempty'}, 3);
    if strcmp(distance, 'sqeuclidean') % compatability for matlab's built in kmeans
        distance = 'euclidean';
    end
    if exist('data', 'var') && ~isempty(data)
        if ~isfloat(data)
            data = double(data);
        end
    end

    model = struct();
    model.clusterer = clusterer;
    model.k = K;
    model.distance = distance;

    if strcmp(clusterer, 'kmeans')
        if isstruct(init) % init with a model
            model = init;
            assert(K == size(init.mu, 2));
        elseif isnumeric(init)
            if size(init, 1) == size(data, 1) && size(init, 2) == 1 % list of category labels
                n_specified_means = numel(unique(init));
                model.mu = zeros(K, size(data, 2));
                for i = 1 : n_specified_means
                    model.mu(i,:) = mean(data(init==i,:), 1);
                end
                randPointIdxs = randperm(size(data, 1), K);
                for i = n_specified_means+1 : K % randomly populate any unspecified means
                    model.mu(i,:) = data(randPointIdxs(i),:);
                end
            elseif size(init, 1) == K && size(init, 2) == size(data, 2) % list of initial means
                model.mu = init;
            else
                error('invalid param init');
            end
        elseif strcmp(init, 'supervised')
            model.mu = InitKMeansSupervised(data, K, labels);
        elseif strcmp(init, 'random') || strcmp(init, 'sample') % sample is for compatability with matlab's built in kmeans
            model.mu = InitKMeansRandomly(data, K);
        elseif strcmp(init, 'furthestfirst')
            model.mu = InitKMeansFurthestFirst(data, K, distance);
        elseif strcmp(init, 'kmeans++') || strcmp(init, 'plus') % plus is for compatability with matlab's built in kmeans
            model.mu = InitKMeansPlusPlus(data, K, distance);
        else
            error('invalid param init');
        end
        
        assert(~any(isnan(model.mu(:))), 'model better not contain NaN!');
        assert(size(model.mu, 1) == K && size(model.mu, 2) == size(data, 2));
        
        if strcmp(distance, 'cosine')
            model.mu = [model.mu,zeros(size(model.mu, 1), 1)];
        end
    elseif strcmp(clusterer, 'gmm')
        if isstruct(init) % init with a model
            model = init;
        elseif strcmp(init, 'random') || strcmp(init, 'sample') % sample is for compatability with matlab's built in kmeans
            [~,model] = ml.GMM(data, K, 1, 'random');
        elseif all(size(init)==[1,size(data, 1)])  % init with labels
            [~,model] = ml.GMM(data, K, 1, init);
        else
            error('invalid param init');
        end
        
        assert(K == size(init.mu, 2));
        assert(~any(isnan(init.mu(:))) && ~any(isnan(init.sigma(:))) && ~any(isnan(init.w(:))), 'model better not contain NaN!');
        assert(~strcmp(distance, 'cosine'));
    elseif startsWith(clusterer, 'hierarchical')
        % nothing to do
    elseif startsWith(clusterer, 'spectralkmeans')
        model.init = init; % we'll do this later
    else
        error('unexpected clusterer');
    end
end


% INPUTS:
%   data - dimensionality of [#datapts][#dims]
%   K - scalar (numeric) num clusters
%   labels
function centroids = InitKMeansSupervised(data, K, labels)
    [N,D] = size(data);
    lSet = unique(labels);
    n_labels = numel(lSet);
    counts = zeros(n_labels, 1);
    classMeans = zeros(n_labels, D);

    for i = 1 : N
        k = find(lSet == labels(i));

        counts(k) = counts(k) + 1;
        classMeans(k,:) = classMeans(k,:) + data(i,:);
    end

    for k = 1 : n_labels
        classMeans(k,:) = classMeans(k,:) ./ counts(k);
    end
    if K > n_labels
        error('We don''t currently support KMeans with supervised labels if there are less category labels than K');
%         centroids = SimpleMeans(data, pts, labels);
    elseif K == n_labels
        centroids = classMeans;
    else % K < n_labels
        error('We don''t currently support KMeans with supervised labels if there are more category labels than K');
%         clustLbls = OldSimpleMeans(classMeans);
%         countsK = zeros(K, 1);
%         centroids = zeros(K, D);
% 
%         for i = 1 : N
%             l = labels(i);
%             int k = Vec.Find(lSet, l);
%             for j = 1 : D
%                 m_centroids(clustLbls(k),j) = centroids(clustLbls(k),j) + data(i,j);
%             end
%             countsK(clustLbls(k)) = countsK(clustLbls(k)) + 1;
%         end
% 
%         for k = 1 : K
%             centroids(k,:) ./ countsK(k);
%         end
    end
end


% set each centroid to a randomly selected data point (selected without replacement)
% INPUTS:
%   data - n_pts x n_dims
%   K - number of clusters
function centroids = InitKMeansRandomly(data, K)
    usablePts = find(sum(data, 2) ~= 0);
    N = numel(usablePts);
    assert(N >= K, 'must have enough points to actually pick K of them');

    randPointIdx = randperm(N, K); % random sample without replacement
%     m_centroids = zeros(K, size(data, 2));
%     for i = 1:K
%         m_centroids(i,:) = data(usablePts(randPointIdx(i)),:);
%     end
    centroids = data(usablePts(randPointIdx),:);
end


% "The standard furthest first algorithm starts with a randomly chosen center and iteratively adds the next center by
% finding the point that has the largest minimum distance to all previously selected centers" (Turnbull & Elkan)
% Hochbaum, D. S., & Shmoys, D. B. (1985). A best possible heuristic for the k-center problem. Mathematics of Operations Research
% Using too many data points for initialization actually makes things WORSE!!! (Fast recognition of musical genres using RBF networks, Turnbull & Elkan 2005)
% This is basically an approximation of the "Metric k-center" problem", which is NP-complete, but desiring longer distance not shorter.
% INPUTS:
%   data - n_pts x n_dims
function centroids = InitKMeansFurthestFirst(data, K, distance)
    N = size(data, 1);
    if strcmp(distance, 'euclidean')
        distance = 'squaredeuclidean'; % identical result, faster
    end

    firstCenter = randi(N);

    centroidIdx = FurthestFirst1(data, K, firstCenter, distance);
    centroids = data(centroidIdx,:); % K x D
end


% Farthest-First Traversal Algorithm as a 2-approximation for kcenter clustering
% INPUTS:
%   X -  N x N input matrix X is interpreted as the distance matrix for N points in an arbitrary metric space.
%   K - the number of centers to be chosen.
%   L0 - the first centroid index.
% REFERENCE: T.F. Gonzalez. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38:293-306, 1985.
% select landmark points by greedy optimisation:
% Given a set of N points, selects a subset of n points called 'landmark' points by an interative greedy optimisation.
% Specifically, when j landmark points have been chosen, the (j+1)-st landmark point
%   maximises the function 'minimum distance to an existing landmark point'.
% The initial landmark point is arbitrary, and may be chosen randomly or by decree.
% The input data can belong to one of the following types:
%    'metric': N-by-N matrix of distances
% The output is a list of indices for the landmark points, presented in the order of discovery.
% Plex Metric Data Toolbox version 2.5 by Vin de Silva, Patrick Perry and contributors. See PX_PLEXINFO for credits and licensing information.
% Released with Plex version 2.5. [2006-Jul-14]
% Modified by Eli Bowen from http://math.stanford.edu/~yuany/pku/matlab/kcenter.m
function choices = FurthestFirst1(data, K, firstCenter, distance)
    N = size(data, 1);

    choices = zeros(K, 1);
    choices(1) = firstCenter;

    % generate remaining landmarks
    DD = zeros(K, N);
    DD(1,:) = pdist2(data(choices(1),:), data, distance);

    DDmin = DD(1,:); % 1 x N
    for a = 2 : K
        [~,newChoice] = max(DDmin); % find furthest point
        choices(a) = newChoice(1); % in case there were multiple equidistant furthest points
        DD(a,:) = pdist2(data(choices(a),:), data, distance);

%         for i = 1 : numel(DDmin)
%             DDmin(i) = min(DDmin(i), DD(a,i));
%         end
        DDmin = min(DDmin, DD(a,:)); % there are 2 ways to do this: distance from closest previous choice, or distance from mean of previous choices
    end
end


% copied from apache commons math and translated by Eli Bowen
% INPUTS:
%   data     - n_pts x n_dims
%   K        - scalar (numeric) num clusters
%   distance - char
function centroids = InitKMeansPlusPlus(data, K, distance)
    N = size(data, 1);
    assert(N >= K); % must have enough points to actually pick K of them

    taken = false(N, 1); % for each element of pointList, is it no longer available?

    % choose one center uniformly at random from among the data points
    firstPointIndex = randi(N);
    firstPoint = data(firstPointIndex,:);
    centroids = firstPoint; % the resulting list of initial centers
    taken(firstPointIndex) = true; % must mark it as taken
    
    % compute minimum distance squared of elements of pointList to elements of resultSet
    if strcmp(distance, 'euclidean') || strcmp(distance, 'squaredeuclidean') % just to save compute time
        minDistSquared = pdist2(firstPoint, data, 'squaredeuclidean');
    else
        minDistSquared = pdist2(firstPoint, data, distance) .^ 2;
    end

    % initialize the elements. Since the only point in resultSet is firstPoint, this is very easy
    ptNums = 1:N;
    ptNums(ptNums == firstPointIndex) = []; % that point isn't considered

    while size(centroids, 1) < K
        distSqSum = sum(minDistSquared(~taken)); % sum of the squared distances for the points in pointList not already taken

        r = rand() * distSqSum; % add one new data point as a center. Each point x is chosen with probability proportional to D(x)2

        nextPointIndex = -1; % index of the next point to be added to the resultSet

        % sum through the squared min distances again, stopping when sum >= r
        tempSum = 0.0;
        for i = 1 : N
            if ~taken(i)
                tempSum = tempSum + minDistSquared(i);
                if tempSum >= r
                    nextPointIndex = i;
                    break
                end
            end
        end

        % if it's not set to >= 0, the point wasn't found in the previous for loop, probably because distances are extremely small. Just pick the last available point.
        if nextPointIndex == -1
            for i = N:-1:1
                if ~taken(i)
                    nextPointIndex = i;
                    break
                end
            end
        end

        % we found one
        if nextPointIndex >= 0
            p = data(nextPointIndex,:);
            centroids = [centroids;p];
            taken(nextPointIndex) = true; % mark it as taken
            ptNums(ptNums == nextPointIndex) = []; % only have to worry about the points still not taken

            if size(centroids, 1) < K
                % now update elements of minDistSquared. We only have to compute the distance to the new center to do this
                if strcmp(distance, 'euclidean') || strcmp(distance, 'squaredeuclidean') % just to save compute time
                    d = pdist2(p, data(ptNums,:), 'squaredeuclidean');
                else
                    d = pdist2(p, data(ptNums,:), distance) .^ 2;
                end
%                 for i = 1:numel(ptNums)
%                     if d(i) < minDistSquared(ptNums(i))
%                         minDistSquared(ptNums(i)) = d(i);
%                     end
%                 end
                minDistSquared(ptNums) = min(minDistSquared(ptNums), d); % pairwise minimums
            end
        else
            break % none found - break to prevent an infinite loop
        end
    end
end