% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% render 10 most category-specific and 50 least category-specific components, along with the activations of the datapoint that each best matches
% INPUTS
%   path - (char) output directory
%   model
%   dat
%   code
%   bank - (char) component bank to render
function [] = RenderComponentBestMatches(path, model, dat, code, bank)
    arguments
        path(1,:) char, model(1,1) Model, dat(1,1) Dataset, code(1,1) struct, bank(1,:) char
    end
    do_pretty = false;

    t = tic();

    if isempty(dat.img_sz) || ~any(strcmp(model.tier1_compbank_names, bank)) % non-image or non-tier-1
        [row,col] = geom.FindCircleCoords(dat.n_nodes);
        row = row .* 22 + 3;
        col = col .* 22 + 3;
        imgSz = [28,28,1];
        lineWidth = 2; % in pixels
    else
        [row,col] = PixelRowCol(dat.img_sz);
        imgSz = dat.img_sz;
        lineWidth = 8; % in pixels
    end
    
    tier1Bank = model.tier1_compbank_names{1};
    
    n_rows = 6; n_cols = 10;
    img = [];
    
    % select 10 highly selective features
    score = max(code.hist.(bank), [], 1) ./ sum(code.hist.(bank), 1); % 1 x n_feats
    if sum(score == max(score)) > 10 % lotsa ties
        comp2RenderIdx = find(score == max(score));
        comp2RenderIdx = comp2RenderIdx(randperm(numel(comp2RenderIdx), 10));
    else
        [~,comp2RenderIdx] = maxk(score, 10);
    end
    comp2RenderPremergeIdx = code.premerge_idx.(bank)(comp2RenderIdx,1); % convert post-merge feature idx back to pre-merge feature idx
    for i = 1 : min(numel(comp2RenderPremergeIdx), 10)
        currImg = PlotGraph(model.compbanks.(tier1Bank).edge_states(:,comp2RenderPremergeIdx(i)), code.hist.(bank)(:,comp2RenderIdx(i)), row, col, model.compbanks.(tier1Bank).edge_endnode_idx, imgSz, [], true, true, dat.node_name, do_pretty, [], lineWidth);
        if isempty(img)
            outImgSz = size(currImg);
            img = ones(n_rows * outImgSz(1), n_cols * outImgSz(2) + 48, 3);
        end
        img(1:outImgSz(1),(i-1)*outImgSz(1) + (1:outImgSz(2)),:) = currImg;
%         if i == 1
%             ylabel('highly selective');
%         end
    end
    
    img = insertText(img, [0,outImgSz(1) + 1], 'highest selectivity', 'BoxOpacity', 0);
    img(outImgSz(1) + 16 + (1:16),:,:) = 0; % black dividing line
    img = insertText(img, [0,outImgSz(1) + 33], 'lowest selectivity', 'BoxOpacity', 0);

    % select 50 many-category (partially non-selective) features
    score = max(code.hist.(bank), [], 1) ./ sum(code.hist.(bank), 1);  % n_feats x 1
    if sum(score == min(score)) > 50 % lotsa ties
        comp2RenderIdx = find(score == min(score));
        comp2RenderIdx = comp2RenderIdx(randperm(numel(comp2RenderIdx), 50));
    else
        [~,comp2RenderIdx] = mink(score, 50);
    end
    comp2RenderPremergeIdx = code.premerge_idx.(bank)(comp2RenderIdx,1); % convert post-merge feature idx back to pre-merge feature idx
    renderRow = 1; % 0-based indexing
    renderCol = 0; % 0-based indexing
    for i = 1 : min(numel(comp2RenderPremergeIdx), 50)
        currImg = PlotGraph(model.compbanks.(tier1Bank).edge_states(:,comp2RenderPremergeIdx(i)), code.hist.(bank)(:,comp2RenderIdx(i)), row, col, model.compbanks.(tier1Bank).edge_endnode_idx, imgSz, [], true, true, dat.node_name, do_pretty, [], lineWidth);
        img(renderRow*outImgSz(1) + 48 + (1:outImgSz(1)),renderCol*outImgSz(1) + (1:outImgSz(2)),:) = currImg;
        renderRow = renderRow + floor((renderCol+1) / n_cols);
        renderCol = mod(renderCol+1, n_cols);
    end

    fig.print(img, path, ['compbestmatch_',bank,'.png']);

    Toc(t);
end