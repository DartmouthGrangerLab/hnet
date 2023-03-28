% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   path      - (char) output directory
%   model     - (Model)
%   dat       - (Dataset)
%   code      - (struct)
%   pt2Render - scalar (numeric index)
%   append    - scalar (string) text to append to file name
function [] = RenderDatapoint(path, model, dat, code, pt2Render, append)
    arguments
        path(1,:) char, model(1,1) Model, dat(1,1) Dataset, code(1,1) struct, pt2Render(1,1), append(1,1) string
    end
    do_pretty = false;
    
    if isempty(dat.img_sz)
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

    n_worst_comps_per_img = 5;
    n_best_comps_per_img = 15;
    pixelValues = dat.pixels(:,pt2Render);
    hist = code.hist.(model.output_bank_name);
    tier1Bank = model.tier1_compbank_names{1};

    tier1_edge_endnode_idx = model.compbanks.(tier1Bank).edge_endnode_idx; % slow functionality
    
    n_rows = floor(sqrt(n_worst_comps_per_img+n_best_comps_per_img)); n_cols = ceil(sqrt(n_worst_comps_per_img+n_best_comps_per_img));
    img = [];
    
    % find worst components for this datapoint
    [~,comp2RenderIdx] = mink(code.comp_code.(model.output_bank_name)(:,pt2Render), n_worst_comps_per_img); % best matching output bank components
    if model.n_compbanks == 1 % just render the only component bank
        comp2RenderPremergeIdx = comp2RenderIdx;
    elseif model.n_compbanks == 2
        comp2RenderPremergeIdx = code.premerge_idx.(model.output_bank_name)(comp2RenderIdx,pt2Render); % convert tier 2 (post-merge) component idx back to tier 1 (pre-merge) component idx
    else
        return
    end
    
    [~,idx] = sort(code.comp_code.(tier1Bank)(comp2RenderPremergeIdx,pt2Render)); % sort the selected components by energy
    comp2RenderIdx = comp2RenderIdx(idx);
    comp2RenderPremergeIdx = comp2RenderPremergeIdx(idx);
    
    for i = 1 : min(numel(comp2RenderIdx), n_worst_comps_per_img)
        [~,idx] = max(hist(:,comp2RenderIdx(i)));
        
        plotTitle = ['cls="',dat.uniq_classes{idx},'", cmp-pt energy=',num2str(code.comp_code.(tier1Bank)(comp2RenderPremergeIdx(i),pt2Render)),', pt energy range=',num2str(min(code.comp_code.(model.output_bank_name)(:,pt2Render))),'-',num2str(max(code.comp_code.(model.output_bank_name)(:,pt2Render)))];
        
        currImg = PlotGraph(model.compbanks.(tier1Bank).edge_states(:,comp2RenderPremergeIdx(i)), hist(:,comp2RenderIdx(i)), row, col, tier1_edge_endnode_idx, imgSz, pixelValues, true, true, model.compbanks.(tier1Bank).g.node_metadata.name, do_pretty, plotTitle, lineWidth);
        if isempty(img)
            outImgSz = size(currImg);
            img = ones(n_rows * outImgSz(1), n_cols * outImgSz(2) + 48, 3);
        end
        img(1:outImgSz(1), (i-1)*outImgSz(1) + (1:outImgSz(2)),:) = currImg;
    end
    
    img = insertText(img, [0,outImgSz(1) + 1], "worst", BoxOpacity=0);
    img(outImgSz(1) + 16 + (1:16),:,:) = 0; % black dividing line
    img = insertText(img, [0,outImgSz(1) + 33], "best", BoxOpacity=0);
    
    % find best components for this image
    [~,comp2RenderIdx] = maxk(code.comp_code.(model.output_bank_name)(:,pt2Render), n_best_comps_per_img); % best matching output bank components
    if model.n_compbanks == 1 % just render the only component bank
        comp2RenderPremergeIdx = comp2RenderIdx;
    elseif model.n_compbanks == 2
        comp2RenderPremergeIdx = code.premerge_idx.(model.output_bank_name)(comp2RenderIdx,pt2Render); % convert post-merge feature idx back to pre-merge feature idx
    else
        return
    end
    
    [~,idx] = sort(code.comp_code.(tier1Bank)(comp2RenderPremergeIdx,pt2Render)); % sort the selected components by energy
    comp2RenderIdx = comp2RenderIdx(idx);
    comp2RenderPremergeIdx = comp2RenderPremergeIdx(idx);
    
    renderRow = 1; % 0-based indexing
    renderCol = 0; % 0-based indexing
    for i = 1 : min(numel(comp2RenderIdx), n_best_comps_per_img)
        [~,idx] = max(hist(:,comp2RenderIdx(i)));
        plotTitle = ['cls="',dat.uniq_classes{idx},'", cmp-pt energy=',num2str(code.comp_code.(tier1Bank)(comp2RenderPremergeIdx(i),pt2Render)),', pt energy range=',num2str(min(code.comp_code.(model.output_bank_name)(:,pt2Render))),'-',num2str(max(code.comp_code.(model.output_bank_name)(:,pt2Render)))];
        
        currImg = PlotGraph(model.compbanks.(tier1Bank).edge_states(:,comp2RenderPremergeIdx(i)), hist(:,comp2RenderIdx(i)), row, col, tier1_edge_endnode_idx, imgSz, pixelValues, true, true, model.compbanks.(tier1Bank).g.node_metadata.name, do_pretty, plotTitle, lineWidth);
        img(renderRow*outImgSz(1) + 48 + (1:outImgSz(1)),renderCol*outImgSz(1) + (1:outImgSz(2)),:) = currImg;
        renderRow = renderRow + floor((renderCol+1) / n_cols);
        renderCol = mod(renderCol+1, n_cols);
    end
    
    fig.print(img, path, char(append + "_edgesim.png"));
end