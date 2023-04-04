% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% 2d scatter plot for all neurons - discrim vs shared
% INPUTS
%   path   - (char) output directory
%   model  - (Model)
%   code   - (struct)
%   dat    - (Dataset)
%   append - (char)
function [] = RenderDiscrimVsSharednessVsFrequency(path, model, code, dat, append)
    arguments
        path(1,:) char, model(1,1) Model, code(1,1) struct, dat(1,1) Dataset, append(1,:) char
    end
    do_pretty = false;
    
    t = tic();
    
    dpi = 150;
    if do_pretty
        dpi = 300;
    end
    
    hist = code.hist.(model.output_bank_name);
    tier1Bank = 'connectedpart'; % TODO: ask model what the bank's name is

    [discriminability,sharedness] = stat.SparseClassResponderDiscriminability(hist);
    nanMsk = isnan(discriminability);
    
    frequency = sum(code.comp_code.(model.output_bank_name), 2);
    frequency = frequency ./ max(frequency);
    
    idx = zeros(2, 9);
    [~,idx(:,1)] = maxk(discriminability, 2);                   discriminability(idx(:,1)) = NaN; sharedness(idx(:,1)) = NaN; % never choose again
    [~,idx(:,2)] = mink(discriminability, 2);                   discriminability(idx(:,2)) = NaN; sharedness(idx(:,2)) = NaN; % never choose again
    [~,idx(:,3)] = mink(abs(discriminability-0.5), 2);          discriminability(idx(:,3)) = NaN; sharedness(idx(:,3)) = NaN; % never choose again
    [~,idx(:,4)] = maxk(sharedness, 2);                         discriminability(idx(:,4)) = NaN; sharedness(idx(:,4)) = NaN; % never choose again
    [~,idx(:,5)] = mink(sharedness, 2);                         discriminability(idx(:,5)) = NaN; sharedness(idx(:,5)) = NaN; % never choose again
    [~,idx(:,6)] = mink(abs(sharedness-0.5), 2);                discriminability(idx(:,6)) = NaN; sharedness(idx(:,6)) = NaN; % never choose again
    [~,idx(:,7)] = maxk(discriminability+sharedness, 2);        discriminability(idx(:,7)) = NaN; sharedness(idx(:,7)) = NaN; % never choose again
    [~,idx(:,8)] = mink(discriminability+sharedness, 2);        discriminability(idx(:,8)) = NaN; sharedness(idx(:,8)) = NaN; % never choose again
    [~,idx(:,9)] = mink(abs(discriminability+sharedness-1), 2); discriminability(idx(:,9)) = NaN; sharedness(idx(:,9)) = NaN; % never choose again
    
    h = figure(Visible="off", defaultAxesFontSize=18); % default = 10
    scatter3(discriminability(~nanMsk), sharedness(~nanMsk), frequency(~nanMsk)');
    for i = 1 : numel(idx)
        text(discriminability(idx(i)), sharedness(idx(i)), frequency(idx(i)), num2str(idx(i)));
    end
    ax = gca();
    ax.XLim = [0,1];
    ax.YLim = [0,1];
    ax.ZLim = [0,1];
    xlabel("discriminability"); ylabel("sharedness"); zlabel("relative frequency");
    box on
    grid on
    view(135, 45);
    fig.print(h, Config.OUT_DIR, char(['discrimvsshared_',append]), [10,10]);

    if ~isempty(dat.img_sz)
        [row,col] = PixelRowCol(dat.img_sz);
        lineWidth = 8; % in pixels

        h = figure(Visible="off", defaultAxesFontSize=18); hold on; % default = 10
        n_rows = 3; n_cols = 6; margin = [0.05,0.05];

        feat2RenderIdx = idx(:);
        if isfield(model.compbanks, "group") % if we have groups
            error("TODO");
        else
            feat2RenderPremergeIdx = feat2RenderIdx;
        end

        tier1_edge_endnode_idx = model.compbanks.(tier1Bank).edge_endnode_idx; % slow functionality
        for i = 1 : numel(feat2RenderIdx)
            fig.subplot(n_rows, n_cols, i, margin);
            PlotGraph(model.compbanks.(tier1Bank).edge_states(:,feat2RenderPremergeIdx(i)), hist(:,feat2RenderIdx(i)), row, col, tier1_edge_endnode_idx, dat.img_sz, [], true, false, dat.node_name, do_pretty, [], lineWidth);
            ax = gca();
            ax.XTick = 1:dat.n_classes;
            ax.XTickLabels = dat.uniq_classes;
            ax.YTick = 0;
            xlabel(num2str(feat2RenderIdx(i))); ylabel("histogram count");
        end

        fig.print(h, path, char(['discrimvsshared_examples_',append]), 'auto', dpi);
    end
    
    Toc(t);
end