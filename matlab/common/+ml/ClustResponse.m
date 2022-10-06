% Eli Bowen 12/6/16
% performs an element-wise transform on each input in dist (changes distance to similarity via some function I really like at the moment)
% INPUTS
%   model - struct
%   data
% RETURNS
%   data
% see also ml.ClustInit, ml.Cluster
function data = ClustResponse(model, data)
    if strcmp(model.clusterer, 'kmeans')
        if iscell(model.mu) % bagging was used
            projecteddata = [];
            for i = 1 : numel(model.mu)
                bagModel = model;
                bagFeats = model.mu{i}.bagFeats;
                bagModel.mu = model.mu{i}.bagComponents;
                projecteddata = [projecteddata,DistFromClustKMeans(bagModel, data(:,bagFeats))];
            end
            data = projecteddata;
        else
            data = ml.DistFromClustKMeans(model, data);
        end

        if strcmp(model.distance, 'euclidean')
            data = 1 ./ (1+data);
        elseif strcmp(model.distance, 'cosine') || strcmp(model.distance, 'jaccard')
            data = 1 - data;
        elseif strcmp(model.distance, 'correlation')
            data = 2 - data;
        else
            error('invalid param distance');
        end
    elseif strcmp(model.clusterer, 'gmm')
        data = ml.DistFromClustGMM(model, data);
        
        data = 1 - data;
    elseif startsWith(model.clusterer, 'hierarchical')
        error('n/a');
    elseif startsWith(model.clusterer, 'spectralkmeans')
        error('TODO');
    else
        error('unknown clusterer');
    end
end