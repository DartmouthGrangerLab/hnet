% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   path      - (char) output directory
%   model
%   dat
%   code
%   bank      - (char) name of bank to render
%   do_pretty - scalar (logical)
%   do_nodes  - scalar (logical)
function [] = RenderAll(path, model, dat, code, bank, do_pretty, do_nodes)
    arguments
        path(1,:) char, model(1,1) Model, dat(1,1) Dataset, code(1,1) struct, bank(1,:) char, do_pretty(1,1) logical, do_nodes(1,1) logical
    end
    max_to_print = 100;
    do_img = true;
    
    t = tic();
    
    if do_pretty
        append = "_pretty.png";
        dpi = 300;
    else
        append = ".png";
        dpi = 150;
    end

    [~,inNodeNames] = inedges(model.g, bank);
    inputBank = inNodeNames{1}; %TODO: should support multiple input banks

    for i = 1 : min(max_to_print, model.compbanks.(bank).n_cmp) % for each component in this bank
        if ~do_img
            h = figure('Visible', 'off', 'defaultAxesFontSize', 14); % default = 10
        end

        if isempty(dat.img_sz) || ~strcmp(inputBank, "sense") % non-image or non-tier-1
            [row,col] = geom.FindCircleCoords(model.compbanks.(bank).g.n_nodes);
            row = row .* 22 + 3;
            col = col .* 22 + 3;
            imgSz = [28,28,1];
            lineWidth = 2; % in pixels
        else
            [row,col] = PixelRowCol(dat.img_sz);
            imgSz = dat.img_sz;
            lineWidth = 8; % in pixels
        end
        img = PlotGraph(model.compbanks.(bank).edge_states(:,i), code.hist.(bank)(:,i), row, col, model.compbanks.(bank).edge_endnode_idx, imgSz, [], true, do_img, model.compbanks.(bank).g.node_metadata.name, do_pretty, ['cmp ',bank,'.',num2str(i)], lineWidth, do_nodes);

        if ~all(strcmp(inNodeNames, "sense")) % this is a higher tier component bank
            img = cat(2, img, zeros(size(img, 1), 16, 3)); % draw black bar separator
            
            % render the upstream components that are used in the most edges
            [~,temp] = inedges(model.g, inputBank);
            if isempty(dat.img_sz) || ~strcmp(temp{1}, "sense") % non-image or non-tier-1
                [row,col] = geom.FindCircleCoords(model.compbanks.(inputBank).g.n_nodes);
                row = row .* 22 + 3;
                col = col .* 22 + 3;
                imgSz = [28,28,1];
                lineWidth = 2; % in pixels
            else
                [row,col] = PixelRowCol(dat.img_sz);
                imgSz = dat.img_sz;
                lineWidth = 8; % in pixels
            end
            mask = (model.compbanks.(bank).edge_states(:,i) ~= EDG.NULL);
            useCount = CountNumericOccurrences(reshape(model.compbanks.(bank).edge_endnode_idx(mask,:), 1, []), 1:model.compbanks.(bank).g.n_nodes);
            [~,idx] = sort(useCount, "descend");
            idx(6:end) = []; % just print the top 5 nodes
            idx(idx == 0) = []; % never print unused nodes
            for j = idx
                currImg = PlotGraph(model.compbanks.(inputBank).edge_states(:,j), code.hist.(inputBank)(:,j), row, col, model.compbanks.(inputBank).edge_endnode_idx, imgSz, [], true, do_img, model.compbanks.(inputBank).g.node_metadata.name, do_pretty, char("cmp "+inputBank+"."+num2str(j)+", used in "+num2str(useCount(j))+" edges"), lineWidth, do_nodes);
                img = cat(2, img, currImg);
            end
        end
        
        if do_img
            fig.print(img, path, char(bank + "_cmp" + num2str(i) + append));
        else
            fig.print(h, path, char(bank + "_cmp" + num2str(i) + append), [4,4], dpi);
        end
    end
    
    Toc(t);
end