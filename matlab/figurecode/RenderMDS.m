% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   path - (char) output directory
%   dat
%   model
%   code
%   inPredLabelIdx
%   edgePredLabelIdx
%   codePredLabelIdx
%   distMeasure - (char) input to pdist
%   mode        - (char) 'scatter' | 'numbers' | '+/-'
%   append      - (char) text to append to output file names
function [] = RenderMDS(path, dat, model, code, inPredLabelIdx, edgePredLabelIdx, codePredLabelIdx, distMeasure, mode, append)
    arguments
        path(1,:) char, dat(1,1) Dataset, model(1,1) Model, code(1,1) struct, inPredLabelIdx, edgePredLabelIdx, codePredLabelIdx, distMeasure(1,:) char, mode(1,:) char, append(1,:) char
    end
    senseDidx = NeighborPairs(model.compbanks.(model.tier1_compbank_names{1}).graph_type, dat.n_nodes, model.compbanks.(model.tier1_compbank_names{1}).imgsz);
    
    t = tic();
    
    h = figure(Visible="off", defaultAxesFontSize=12); % default = 10
    n_rows = 2 + model.n_compbanks; n_cols = 2; margin = [0.06,0.06];
    
    pDist = squareform(pdist(double(dat.pixels'), distMeasure));
    
    fig.subplot(n_rows, n_cols, 1, margin);
    warning("off", "stats:pca:ColRankDefX");
    [~,score,~,~,explained,~] = pca(double(dat.pixels'), NumComponents=3);
    Helper(score, dat.label_idx, inPredLabelIdx, dat.n_classes, mode);
    xlabel(['pc 1 (',num2str(explained(1), '%.2f'),'% expl.)']);
    ylabel(['pc 2 (',num2str(explained(2), '%.2f'),'% expl.)']);
    zlabel(['pc 3 (',num2str(explained(3), '%.2f'),'% expl.)']);
    title("pca(input 'pixels')", FontSize=10); % default = 11

    fig.subplot(n_rows, n_cols, 2, margin);
    mds = cmdscale(pDist, 3);
    Helper(mds, dat.label_idx, inPredLabelIdx, dat.n_classes, mode);
    title("cmds(input 'pixels')", FontSize=10); % default = 11
    
    edges = Edge2Logical(GetEdgeStates(dat.pixels, senseDidx, [EDG.NCONV,EDG.NIMPL,EDG.AND]));
    pDist = squareform(pdist(double(edges'), distMeasure));
    
    fig.subplot(n_rows, n_cols, 3, margin);
    [~,score,~,~,explained,~] = pca(double(edges'), NumComponents=3);
    Helper(score, dat.label_idx, edgePredLabelIdx, dat.n_classes, mode);
    xlabel(['pc 1 (',num2str(explained(1), '%.2f'),'% expl.)']);
    ylabel(['pc 2 (',num2str(explained(2), '%.2f'),'% expl.)']);
    zlabel(['pc 3 (',num2str(explained(3), '%.2f'),'% expl.)']);
    title("pca(edges)", FontSize=10); % default = 11

    fig.subplot(n_rows, n_cols, 4, margin);
    mds = cmdscale(pDist, 3);
    Helper(mds, dat.label_idx, edgePredLabelIdx, dat.n_classes, mode);
    title("cmds(edges)", FontSize=10); % default = 11
    
    for i = 1 : model.n_compbanks
        bank = model.compbank_names{i};

        fig.subplot(n_rows, n_cols, 4 + 2*(i-1) + 1, margin);
        [~,score,~,~,explained,~] = pca(double(code.comp_code.(bank)'), NumComponents=3);
        Helper(score, dat.label_idx, codePredLabelIdx, dat.n_classes, mode);
        xlabel(['pc 1 (',num2str(explained(1), '%.2f'),'% expl.)']);
        ylabel(['pc 2 (',num2str(explained(2), '%.2f'),'% expl.)']);
        zlabel(['pc 3 (',num2str(explained(3), '%.2f'),'% expl.)']);
        title(['pca(logicnet ',bank,' encoding)'], FontSize=10); % default = 11

        fig.subplot(n_rows, n_cols, 4 + 2*(i-1) + 2, margin);
        try
            pDist = squareform(pdist(double(code.comp_code.(bank)'), distMeasure));
            mds = cmdscale(pDist, 3);
            Helper(mds, dat.label_idx, codePredLabelIdx, dat.n_classes, mode);
        end
        title(['cmds(logicnet ',bank,' encoding)'], FontSize=10); % default = 11
    end

    fig.print(h, path, char(['mds_',append]), 'auto', 300);
    
    Toc(t);
end


function [] = Helper(data, labelIdx, predLabelIdx, n_classes, mode)
    color = linspecer(n_classes, "qualitative");

    data = data - min(data, [], 1);
    data = data ./ max(data, [], 1);

    if mode == "scatter"
        for i = 1 : numel(labelIdx)
            scatter3(data(i,1), data(i,2), data(i,3), "*", MarkerEdgeColor=color(labelIdx(i),:));
        end
        xlabel("X"); ylabel("Y"); zlabel("Z");
        grid on
        box on
        view(135, 45);
    elseif mode == "numbers"
        for i = 1 : 2 : numel(labelIdx)
            text(data(i,1), data(i,2), data(i,3), num2str(i), Color=color(labelIdx(i),:));
        end
        xlabel("X"); ylabel("Y"); zlabel("Z");
        grid on
        box on
        view(135, 45);
    elseif mode == "+/-"
        fig.Plot3DScatterWithShadows(gcf(), data(:,1), data(:,2), data(:,3), [], color(labelIdx,:), '', [], [], [], true, false, true);
        fig.Plot3DScatterWithShadows(gcf(), data(labelIdx(:)==predLabelIdx(:),1), data(labelIdx(:)==predLabelIdx(:),2), data(labelIdx(:)==predLabelIdx(:),3), [], color(labelIdx(labelIdx(:)==predLabelIdx(:)),:), '+', [], [], [], false, true);
        fig.Plot3DScatterWithShadows(gcf(), data(labelIdx(:)~=predLabelIdx(:),1), data(labelIdx(:)~=predLabelIdx(:),2), data(labelIdx(:)~=predLabelIdx(:),3), [], color(labelIdx(labelIdx(:)~=predLabelIdx(:)),:), '_', [], [], [], false, true);
    else
        error("unexpected mode");
    end
    set(gca, "TickLength", [0,0]);
    xLims = xlim();
    yLims = ylim();
    zLims = zlim();
    ax = gca();
    ax.XTick = [0,0.5*xLims(2),xLims(2)];
    ax.YTick = [0,0.5*yLims(2),yLims(2)];
    ax.ZTick = [0,0.5*zLims(2),zLims(2)];
end