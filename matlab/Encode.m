% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   compbank
%   data
%   encodeSpec
% RETURNS
%   compCode - n_cmp x n_pts (numeric)
%   premergeIdx
function [compCode,premergeIdx] = Encode(compbank, data, encodeSpec)
    arguments
        compbank(1,1) ComponentBank, data(:,:), encodeSpec(1,:) char
    end
    validateattributes(data, {'logical','double','single'}, {});
    if compbank.n_cmp == 0
        compCode = [];
        premergeIdx = [];
        return
    end
    n_pts = size(data, 2);
    
    t = tic();

    compCode = data;
    steps = strsplit(encodeSpec, '-->');
    for ii = 1 : numel(steps)
        step = strsplit(steps{ii}, '.');
        task = step{1};

        if strcmp(task, 'energy')
            compCode = Energy(compbank, compCode);
            premergeIdx = repmat((1:compbank.n_cmp)', 1, size(compCode, 2)); % n_groups x n_pts
        elseif strcmp(task, 'max')
            newCompCode = zeros(n_pts, compbank.n_cmp, 'single'); % transposed at the end
            premergeIdx = zeros(n_pts, compbank.n_cmp); % transposed at the end
            for i = 1 : compbank.n_cmp
                idx = find(compbank.edge_states(:,i) ~= EDG.NULL);
                if ~isempty(idx)
                    [newCompCode(:,i),premergeIdx(:,i)] = max(compCode(idx,:), [], 1);
                    premergeIdx(:,i) = idx(premergeIdx(:,i)); % map back to the full list
                end
            end
            compCode = newCompCode';
            premergeIdx = premergeIdx';
        elseif strcmp(task, 'maxabs')
            newCompCode = zeros(n_pts, compbank.n_cmp, 'single'); % transposed at the end
            premergeIdx = zeros(n_pts, compbank.n_cmp); % transposed at the end
            for i = 1 : compbank.n_cmp
                idx = find(compbank.edge_states(:,i) ~= EDG.NULL);
                if ~isempty(idx)
                    [newCompCode(:,i),premergeIdx(:,i)] = maxk(compCode(idx,:), 1, 1, 'ComparisonMethod', 'abs'); % energy furthest from zero
                    premergeIdx(:,i) = idx(premergeIdx(:,i)); % map back to the full list
                end
            end
            compCode = newCompCode';
            premergeIdx = premergeIdx';
        elseif strcmp(task, 'wta')
            n_cmp_per_img = str2double(step{2}); % integer, e.g. 20
            compCode = logical(ml.KWTA(compCode, n_cmp_per_img)); % n_trn x n_cmp_groups
        else
            error('unexpected task');
        end
    end

    Toc(t, toc(t) > 1);
end