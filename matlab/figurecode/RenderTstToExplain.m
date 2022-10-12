% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% INPUTS
%   path      - (char) output directory
%   model
%   dat
%   code
%   bank      - (char) name of bank to render
%   do_pretty - scalar (logical)
function [] = RenderTstToExplain(path, model, dat, code, bank, do_pretty)
    arguments
        path(1,:) char, model(1,1) Model, dat(1,1) Dataset, code(1,1) struct, bank(1,:) char, do_pretty(1,1)
    end
    max_to_print = 20;
    do_img = true;
    
    t = tic();
    
    if do_pretty
        append = '_pretty.png';
        dpi = 300;
    else
        append = '.png';
        dpi = 150;
    end

    [~,inNodeNames] = inedges(model.g, bank);
    if all(strcmp(inNodeNames, 'sense')) % this must be a higher tier component bank
        return
    end

    inputBank = inNodeNames{1}; %TODO: should support multiple input banks

    for i = 1 : min(max_to_print, dat.n_pts) % for each component in this bank
        [nrg,order] = sort(code.comp_code.(bank)(:,i), 'descend'); % find components best matching this datapoint
        if model.compbanks.(bank).n_cmp > 10
            nrg = nrg([1:5,end-4:end]);
            order = order([1:5,end-4:end]);
        end
        for j = 1 : numel(order)
            cmp = order(j);

            if ~do_img
                h = figure('Visible', 'off', 'defaultAxesFontSize', 14); % default = 10
            end
    
            if isempty(dat.img_sz) || ~strcmp(inNodeNames{1}, 'sense') % non-image or non-tier-1
                [row,col] = geom.FindCircleCoords(model.compbanks.(bank).n_nodes);
                row = row .* 22 + 3;
                col = col .* 22 + 3;
                imgSz = [28,28,1];
                lineWidth = 2; % in pixels
            else
                [row,col] = PixelRowCol(dat.img_sz);
                imgSz = dat.img_sz;
                lineWidth = 8; % in pixels
            end
    
            nodeActivations = code.comp_code.(inputBank)(:,i);
            img = PlotGraph(model.compbanks.(bank).edge_states(:,cmp), code.hist.(bank)(:,cmp), row, col, model.compbanks.(bank).edge_endnode_idx, imgSz, nodeActivations, true, do_img, model.compbanks.(bank).node_name, do_pretty, ['dat',num2str(i),'_cmp ',bank,'.',num2str(cmp),'_energy',num2str(nrg(j))], lineWidth);
    
            % render the upstream components that are used in the most edges
            img = cat(2, img, zeros(size(img, 1), 16, 3)); % draw black bar separator
            [~,inputInputBank] = inedges(model.g, inputBank);
            if isempty(dat.img_sz) || ~strcmp(inputInputBank{1}, 'sense') % non-image or non-tier-1
                [row,col] = geom.FindCircleCoords(model.compbanks.(inputBank).n_nodes);
                row = row .* 22 + 3;
                col = col .* 22 + 3;
                imgSz = [28,28,1];
                lineWidth = 2; % in pixels
            else
                [row,col] = PixelRowCol(dat.img_sz);
                imgSz = dat.img_sz;
                lineWidth = 8; % in pixels
            end
            mask = (model.compbanks.(bank).edge_states(:,cmp) ~= EDG.NULL);
            useCount = CountNumericOccurrences(reshape(model.compbanks.(bank).edge_endnode_idx(mask,:), 1, []), 1:model.compbanks.(bank).n_nodes);
            [~,idx] = sort(useCount, 'descend');
            idx(6:end) = []; % just print the top 5 nodes
            idx(idx == 0) = []; % never print unused nodes
            for k = idx
                nodeActivations = code.comp_code.(inputInputBank{1})(:,i);
                currImg = PlotGraph(model.compbanks.(inputBank).edge_states(:,k), code.hist.(inputBank)(:,k), row, col, model.compbanks.(inputBank).edge_endnode_idx, imgSz, nodeActivations, true, do_img, model.compbanks.(inputBank).node_name, do_pretty, ['cmp ',inputBank,'.',num2str(k),', used in ',num2str(useCount(k)),' edges'], lineWidth);
                img = cat(2, img, currImg);
            end
            
            if do_img
                fig.print(img, fullfile(path, 'explain'), ['dat',num2str(i),'_',bank,'_cmp',num2str(cmp),append]);
            else
                fig.print(h, fullfile(path, 'explain'), ['dat',num2str(i),'_',bank,'_cmp',num2str(cmp),append], [4,4], dpi);
            end
        end
    end
    
    Toc(t);
end