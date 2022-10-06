% INPUTS:
%   labels     - 1D (numeric) array of label IDs for each datapoint
%   frac       - scalar (numeric 0 --> 1) fraction of data to keep
%   randStream - OPTIONAL (random number generator stream)
% RETURNS:
%   idx - indices into a random percent of the datapoints for each class
function idx = RandSubsetDataset(labels, frac, randStream)
    validateattributes(labels, {'numeric'}, {}, 1);
    validateattributes(frac, {'numeric'}, {'nonempty','scalar','nonnegative'}, 2);
    labels = labels(:);

    idx = [];
    uniqLabels = unique(labels);
    for k = 1 : numel(uniqLabels)
        catIndices = find(labels == uniqLabels(k));
        N = numel(catIndices);

        if exist('randStream', 'var') && ~isempty(randStream)
            randIdx = randperm(randStream, N);
        else
            randIdx = randperm(N);
        end
        randIdx = randIdx(1:round(N*frac));
        idx = [idx;catIndices(randIdx)];
    end
end