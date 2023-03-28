% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   edgeStates  - n_edges x 1 (EDG enum)
%   hist        - n_classes x 1 (numeric)
%   row         - n_nodes x 1 (numeric)
%   col         - n_nodes x 1 (numeric)
%   didx        - n_edges x 2 (numeric index)
%   imgSz       - 3 x 1 (int-valued numeric) [n_rows,n_cols,n_chan]
%   nodeActivations - OPTIONAL n_nodes x 1 (logical or numeric)
%   do_color    - OPTIONAL scalar (logical) render edges in color (vs all blue)
%   do_img      - OPTIONAL scalar (logical) render to an image instead of the current matlab figure
%   nodeName    - OPTIONAL
%   do_pretty   - OPTIONAL scalar (logical) hide annotations and make the result look pretty
%   plotTitle   - OPTIONAL (char)
%   lineWidth   - OPTIONAL scalar (int-valued numeric)
%   do_nodes    - OPTIONAL scalar (logical)
% RETURNS
%   img - empty unless do_img == true
function img = PlotGraph(edgeStates, hist, row, col, didx, imgSz, nodeActivations, do_color, do_img, nodeName, do_pretty, plotTitle, lineWidth, do_nodes)
    validateattributes(edgeStates, {'EDG'},     {}, 1);
    validateattributes(hist,       {'numeric'}, {}, 2);
    validateattributes(row,        {'numeric'}, {'nonempty'}, 3);
    validateattributes(col,        {'numeric'}, {'nonempty'}, 4);
    validateattributes(didx,       {'numeric'}, {}, 5);
    validateattributes(imgSz,      {'numeric'}, {'nonempty'}, 6);
    if ~exist("nodeActivations", "var") || isempty(nodeActivations)
        nodeActivations = false(numel(row), 1);
    end
    if ~exist("do_color", "var") || isempty(do_color)
        do_color = false;
    end
    if ~exist("do_img", "var") || isempty(do_img)
        do_img = false;
    end
    if ~exist("do_nodes", "var") || isempty(do_nodes)
        do_nodes = true;
    end
    n_classes = numel(hist);
    scaleFactor = 32; % was 3
    if ~exist("lineWidth", "var") || isempty(lineWidth)
        lineWidth = 2;
    end
    row = row(:); % assumed below
    col = col(:);

    if isnumeric(nodeActivations) && any(nodeActivations)
        nodeActivations = nodeActivations ./ max(nodeActivations); % make ranged 0 --> 1
    end

    % edges
    if do_color
        is_limited_color_pallette = false;
        if all(edgeStates == EDG.NULL | edgeStates == EDG.NOR | edgeStates == EDG.NCONV | edgeStates == EDG.NIMPL | edgeStates == EDG.AND)
            is_limited_color_pallette = true;
        end
        if is_limited_color_pallette
            color([EDG.NOR,EDG.NCONV,EDG.NIMPL,EDG.AND],:) = linspecer(4, "qualitative");
        else
            color = linspecer(16);
        end
    else
        color = zeros(16, 3);
        color(:,3) = 1; % all blue
    end
    
    if do_img
        img = ones(scaleFactor .* imgSz(2), scaleFactor .* imgSz(1), 3);
        row = row .* scaleFactor;
        col = col .* scaleFactor;
        if all(row >= 2)
            row = row - 1; % move points away from the edge for prettyness
        end
        if all(col >= 2)
            col = col - 1; % move points away from the edge for prettyness
        end
        row(row < 1) = 1;
        col(col < 1) = 1;
        row(row > imgSz(1)*scaleFactor) = imgSz(1)*scaleFactor;
        col(col > imgSz(2)*scaleFactor) = imgSz(2)*scaleFactor;
        row = round(row);
        col = round(col);
    else
        img = [];
        hold on
        row = max(row) - row + 1; % flip it - matlab figures are upside-down versions of matlab images...
    end
    
    % plot edges belonging to this component
    for i = 1 : 16
        if any(edgeStates == i)
            if is_limited_color_pallette
                if i == EDG.NIMPL
                    img = Helper(img, row, col, fliplr(didx), (edgeStates == i), 'redblue', lineWidth); % red = active node, blue = inactive node
                elseif i == EDG.NCONV
                    img = Helper(img, row, col, didx, (edgeStates == i), 'redblue', lineWidth); % red = active node, blue = inactive node
                elseif i == EDG.AND
                    img = Helper(img, row, col, didx, (edgeStates == i), [0,0.5,0], lineWidth);
                elseif i == EDG.NOR
                    img = Helper(img, row, col, didx, (edgeStates == i), [0,0,0], lineWidth);
                else
                    error("bug");
                end
            else
                img = Helper(img, row, col, didx, (edgeStates == i), color(i,:), lineWidth);
            end
        end
    end
    
    % plot a sort of hacky edge legend
    if ~do_pretty && do_color
        if is_limited_color_pallette
            if do_img
                img = insertText(img, [0,round(1*scaleFactor)], "0  ", TextColor=[0,0,0.8], BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75));
                img = insertText(img, [0,round(1*scaleFactor)], "  1", TextColor=[0.8,0,0], BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75));
                img = insertText(img, [0,round(1*scaleFactor)], "    (NIMPL / NCONV)", TextColor=[0,0,0], BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75));
                if any(edgeStates == EDG.AND)
                    img = insertText(img, [0,round(2*scaleFactor)], "AND", TextColor=[0,0.5,0], BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75));
                end
                if any(edgeStates == EDG.NOR)
                    img = insertText(img, [0,round(2*scaleFactor)], "NOR", TextColor=[0,0,0], BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75));
                end
            else
                text(0, imgSz(2) - 1*1.5, "0  ", Color=[0,0,0.8]);
                text(0, imgSz(2) - 1*1.5, "  1", Color=[0.8,0,0]);
                text(0, imgSz(2) - 1*1.5, "    (NIMPL / NCONV)"); % black
                if any(edgeStates == EDG.AND)
                    text(0, imgSz(2) - 2*1.5, "AND", Color=[0,0.5,0]);
                end
                if any(edgeStates == EDG.NOR)
                    text(0, imgSz(2) - 2*1.5, "NOR", Color=[0,0,0]);
                end
            end
        else
            count = 1;
            for i = 1 : 16
                if any(edgeStates == i)
                    if do_img
                        img = insertText(img, [0,round(count*scaleFactor)], char(EDG(i)), TextColor=color(i,:), BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75));
                    else
                        text(0, imgSz(2) - count*1.5, char(EDG(i)), Color=color(i,:));
                    end
                    count = count + 1;
                end
            end
        end
    end
    
    % prep nodes
    do_node_name = exist('nodeName', 'var') && ~isempty(nodeName) && ~do_pretty && numel(nodeName) <= 128;
    
    nodeMsk = (nodeActivations ~= 0); % show node name if the pixel is active
    if ~do_pretty && ~isempty(didx)
        for i = 1 : numel(nodeName)
            nodeMsk(i) = any(didx(edgeStates ~= EDG.NULL,1) == i) || any(didx(edgeStates ~= EDG.NULL,2) == i);
        end                 
    end
    
    % plot nodes/pixels on top of edges
    if do_nodes
        if do_img
            if ~any(nodeActivations)
                if do_node_name
                    clr = [0.5,0.5,0.5]; % we can draw these in gray for greater node name visibility
                else
                    clr = 'black';
                end
                img = insertShape(img, "FilledCircle", [col(:),row(:),repmat(scaleFactor/5, numel(col), 1)], Color=clr, SmoothEdges=false, Opacity=1);
            elseif isnumeric(nodeActivations)
                img = insertShape(img, "FilledCircle", [col(:),row(:),repmat(scaleFactor/5, numel(col), 1)], Color=nodeActivations(:).*[1,1,1], SmoothEdges=false, Opacity=1);
            else
                mask = nodeMsk(:) & nodeActivations(:) ~= 0; % plot white pixels
                img = insertShape(img, "Circle", [col(mask),row(mask),repmat(scaleFactor/5, sum(mask), 1)], Color="black", SmoothEdges=false, Opacity=1);
                mask = nodeMsk(:) & nodeActivations(:) == 0; % plot black pixels
                img = insertShape(img, "FilledCircle", [col(mask),row(mask),repmat(scaleFactor/5, sum(mask), 1)], Color="black", SmoothEdges=false, Opacity=1);
            end
        else
            if isnumeric(nodeActivations)
                scatter(col, row, 14, nodeActivations(:) .* [1,1,1], 'o', 'filled');
            else
                scatter(col(~nodeActivations), row(~nodeActivations), 14, 'k', 'o', 'filled'); % plot black pixels
                scatter(col(nodeActivations), row(nodeActivations), 14, 'k', 'o'); % plot white pixels (non-filled)
            end
        end
    end
    
    % plot node labels on top of everything
    if do_node_name
        if do_img
            tempCol = col(nodeMsk);
            tempRow = row(nodeMsk);
            img = insertText(img, [tempCol(:),tempRow(:)], nodeName(nodeMsk), BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75), AnchorPoint="LeftCenter");
        else
            text(col(nodeMsk), row(nodeMsk), nodeName(nodeMsk), Interpreter="none");
        end
    end
    
    % class histogram
    if ~do_pretty && ~isempty(hist)
        color = linspecer(n_classes, "qualitative");
        if do_img
            line = zeros(n_classes, 4);
            for c = 1 : n_classes
                line(c,:) = [c*3,(imgSz(1)*scaleFactor)-1,c*3,(imgSz(1)*scaleFactor) - (hist(c)*0.5*imgSz(1)*scaleFactor)]; % n_lines x 4 [col(pt1),row(pt1),col(pt2),row(pt2)]
            end
            img = insertShape(img, "Line", line, Color=color, SmoothEdges=false, LineWidth=3);
        else
            for c = 1 : n_classes
                bar(c, hist(c) * 0.9 * imgSz(1) / 2, FaceColor=color(c,:), EdgeColor="none", FaceAlpha=0.8);
            end
        end
    end
    
    % title
    if ~do_pretty && exist('plotTitle', 'var') && ~isempty(plotTitle)
        if do_img
            img = insertText(img, [round(imgSz(2)*scaleFactor/2),0], plotTitle, BoxOpacity=0, FontSize=ceil(scaleFactor * 0.75), AnchorPoint="CenterTop");
        else
            text(0, round(imgSz(2)/2), plotTitle, Interpreter="none", HorizontalAlignment="center");
        end
    end
    
    % axes
    if ~do_img
        ax = gca();
        xlim([0,imgSz(2)+1]);
        ylim([0,imgSz(1)+1]);
        ax.XTickLabel = [];
        ax.YTickLabel = [];
        axis square
        if do_pretty
            ax.XTick = [];
            ax.YTick = [];
            axis off
        else
            ax.XTick = 1:max(col);
            ax.YTick = 1:max(row);
            box on
        end
    end
end


function img = Helper(img, row, col, didx, temp, color, lineWidth)
    if ~isempty(img) % do_img
        line = [col(didx(:,1)),row(didx(:,1)),col(didx(:,2)),row(didx(:,2))]; % n_lines x 4
        img = fig.insertShape(img, 'Line', line, 'Color', color, 'LineWidth', lineWidth, 'Alpha', temp, 'SmoothEdges', false);
    else
        fig.plot(col(didx'), row(didx'), 'Color', color, 'LineWidth', lineWidth, 'Alpha', temp);
    end
end