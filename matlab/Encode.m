% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   compbank
%   data
%   encodeSpec
% RETURNS
%   compCode - n_cmp x n_pts (numeric)
%   premergeIdx
function [compcode,premergeIdx] = Encode(compbank, data, encodeSpec)
    arguments
        compbank(1,1) ComponentBank, data(:,:), encodeSpec(1,:) char
    end
    validateattributes(data, {'logical','double','single'}, {});
    if compbank.n_cmp == 0
        compcode = [];
        premergeIdx = [];
        return
    end
    n_pts = size(data, 2);
    
    t = tic();

    compcode = data;
    steps = strsplit(encodeSpec, '-->');
    for ii = 1 : numel(steps)
        step = strsplit(steps{ii}, '.');
        task = step{1};

        if task == "energy"
            compcode = Energy(compbank, compcode);
            premergeIdx = repmat((1:compbank.n_cmp)', 1, size(compcode, 2)); % n_groups x n_pts
        elseif task == "max"
            newCompCode = zeros(n_pts, compbank.n_cmp, "single"); % transposed at the end
            premergeIdx = zeros(n_pts, compbank.n_cmp); % transposed at the end
            for i = 1 : compbank.n_cmp
                idx = find(compbank.edge_states(:,i) ~= EDG.NULL);
                if ~isempty(idx)
                    [newCompCode(:,i),premergeIdx(:,i)] = max(compcode(idx,:), [], 1);
                    premergeIdx(:,i) = idx(premergeIdx(:,i)); % map back to the full list
                end
            end
            compcode = newCompCode';
            premergeIdx = premergeIdx';
        elseif task == "maxabs"
            newCompCode = zeros(n_pts, compbank.n_cmp, "single"); % transposed at the end
            premergeIdx = zeros(n_pts, compbank.n_cmp); % transposed at the end
            for i = 1 : compbank.n_cmp
                idx = find(compbank.edge_states(:,i) ~= EDG.NULL);
                if ~isempty(idx)
                    [newCompCode(:,i),premergeIdx(:,i)] = maxk(compcode(idx,:), 1, 1, 'ComparisonMethod', 'abs'); % energy furthest from zero
                    premergeIdx(:,i) = idx(premergeIdx(:,i)); % map back to the full list
                end
            end
            compcode = newCompCode';
            premergeIdx = premergeIdx';
        elseif task == "wta"
            n_cmp_per_img = str2double(step{2}); % integer, e.g. 20
            compcode = logical(ml.KWTA(compcode, n_cmp_per_img)); % n_trn x n_cmp_groups
        else
            error("unexpected task");
        end
    end

    Toc(t, toc(t) > 1);
end