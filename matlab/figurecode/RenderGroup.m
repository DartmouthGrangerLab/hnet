% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   path - (char) output directory
%   model
%   dat
%   code
%   groupIdx
function [] = RenderGroup(path, model, dat, code, groupIdx)
    arguments
        path(1,:) char, model(1,1) Model, dat(1,1) Dataset, code(1,1) struct, groupIdx(1,1)
    end
    do_pretty = false;
    tier1Bank = model.tier1_compbank_names{1};
    
    t = tic();

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
    
    dpi = 150;
    if do_pretty
        dpi = 300;
    end
    
    comp2RenderPremergeIdx = find(model.compbanks.group.edge_states(:,groupIdx));
    
    h = figure('Visible', 'off', 'defaultAxesFontSize', 14); % default = 10
    n_rows = 6; n_cols = 10;

    for j = 1 : min(n_rows*n_cols, numel(comp2RenderPremergeIdx)) % for each feature in this group
        fig.subplot(n_rows, n_cols, j, [0.015,0.015]);

        pixelValues = dat.pixels(:,code.comp_best_img.(tier1Bank)(comp2RenderPremergeIdx(j))); % values for the image that best matches this component
        
        PlotGraph(model.compbanks.(tier1Bank).edge_states(:,comp2RenderPremergeIdx(j)), code.hist.group(:,groupIdx), row, col, model.compbanks.(tier1Bank).edge_endnode_idx, imgSz, pixelValues, true, false, model.compbanks.(tier1Bank).node_name, do_pretty, [], lineWidth);
    end

    fig.print(h, fullfile(path, 'group'), ['group',num2str(groupIdx)], [40,24], dpi);
    
    Toc(t);
end