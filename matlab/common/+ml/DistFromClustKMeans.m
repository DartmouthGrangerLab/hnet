% Eli Bowen 12/6/16
% performs an element-wise transform on each input in dist (changes distance to similarity via some function I really like at the moment)
% INPUTS
%   model
%   data - n_pts x n_dim
% RETURNS
%   data - n_pts x n_prototypes (numeric) distance between each point and each cluster
function data = DistFromClustKMeans(model, data)
    prototypes = model.mu;
    assert(size(data, 2) == size(prototypes, 2));

    if strcmp(model.distance, 'euclidean')
        try
            gpuSignals = gpuArray(data);
            distances = zeros(size(data, 1), size(prototypes, 1));
            for i = 1 : size(prototypes, 1)
                distances(:,i) = gather(sqrt(sum((gpuSignals - repmat(gpuArray(prototypes(i,:)), size(gpuSignals, 1), 1)).^2, 2))); % even faster via GPU
            end
            data = distances;
        catch
            data = bsxfun(@plus, sum(data.*data, 2), sum(prototypes.*prototypes, 2)') - 2*(data*prototypes'); % 4x as fast as pdist2
            data = max(data, 0); % can be within floating point error of 0, but negative
            data = sqrt(data);
%             for i = 1 : size(prototypes, 1)
%                 distances(:,i) = sqrt(sum((data - repmat(prototypes(i,:), size(data, 1), 1)).^2, 2)); % same thing, sometimes 5X faster, sometimes 5X slower (more RAM)
%             end
%             data = distances;
%             data = pdist2(data, prototypes, 'euclidean'); % faster than above
        end
    elseif strcmp(model.distance, 'cosine')
        data = data ./ sqrt(sum(data .* data, 2));
        data(isnan(data)) = 0;
        data = 1 - (data * prototypes');
%         import matlabclusternetworkjavahelper.*;
%         assert(size(data, 1) < 2^31, 'data must have a length less than MAX_INT');
%         data = matlabclusternetworkjavahelper.PDistComputer.PDistCos(data, prototypes, DetermineNumJavaComputeCores());
%         data = pdist2(data, prototypes, 'cosine'); % a few times faster in recent matlab releases
%         l2Norms = sqrt(sum(data .* data, 2));
% 
%         data2 = matlabclusternetworkjavahelper.PDistComputer.PDistCosJBLAS(data, prototypes, DetermineNumJavaComputeCores());
%         data3 = matlabclusternetworkjavahelper.PDistComputer.PDistCos(data, prototypes, DetermineNumJavaComputeCores());
%         data4 = double(matlabclusternetworkjavahelper.PDistComputer.PDistCosJBLASSingle(data, prototypes, DetermineNumJavaComputeCores()));
%         data5 = matlabclusternetworkjavahelper.PDistComputer.PDistCosMTJ(data, prototypes, DetermineNumJavaComputeCores());
%         data6 = matlabclusternetworkjavahelper.PDistComputer.PDistCosMTJSingle(data, prototypes, DetermineNumJavaComputeCores());
    elseif strcmp(model.distance, 'correlation')
        data = data - mean(data, 2);
        data = data ./ sqrt(sum(data .* data, 2));
        data(isnan(data)) = 0;
        data = 1 - (data * prototypes');
    elseif strcmp(model.distance, 'hamming')
        error('TODO');
    else
        error('invalid param distance');
    end
end