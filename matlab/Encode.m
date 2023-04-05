% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   model - scalar (Model)
%   sense - n_nodes x n_pts (logical) e.g. dat.pixels
% RETURNS
%   compCode - struct of n_cmp x n_pts numerics, one field per bank
%   premergeIdx - struct of numerics, one field per bank
function [compcode,premergeidx] = Encode(model, sense)
    arguments
        model(1,1) Model, sense(:,:) {mustBeLogical}
    end
    assert(size(sense, 1) == model.n_sense);
    [compcode,premergeidx] = Helper(model, struct(sense=sense), struct(sense=[]), "sense");
end


function [compcode,premergeidx] = Helper(model, compcode, premergeidx, currSrc)
    [~,dstBanks] = outedges(model.g, currSrc);
    for i = 1 : numel(dstBanks)
        if dstBanks{i} ~= "out"
            assert(~isfield(compcode, dstBanks{i}), "no support for graph cycles");
            [compcode.(dstBanks{i}),premergeidx.(dstBanks{i})] = BankEncode(model.compbanks.(dstBanks{i}), compcode.(currSrc), model.g.Nodes.encode_spec{findnode(model.g, dstBanks{i})});
        end
    end
    for i = 1 : numel(dstBanks)
        if dstBanks{i} ~= "out"
            [compcode,premergeidx] = Helper(model, compcode, premergeidx, dstBanks{i}); % recurse
        end
    end
end


% INPUTS
%   compbank
%   data
%   encodeSpec
% RETURNS
%   compcode - n_cmp x n_pts (numeric)
%   premergeIdx
function [compcode,premergeIdx] = BankEncode(compbank, data, encodespec)
    arguments
        compbank(1,1) ComponentBank, data(:,:), encodespec(1,1) string
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
    steps = strsplit(encodespec, "-->");
    for ii = 1 : numel(steps)
        step = strsplit(steps{ii}, ".");
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
                    [newCompCode(:,i),premergeIdx(:,i)] = maxk(compcode(idx,:), 1, 1, ComparisonMethod="abs"); % energy furthest from zero
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