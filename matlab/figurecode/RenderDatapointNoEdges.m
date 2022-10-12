% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% INPUTS
%   path      - (char) output directory
%   dat       - scalar (Dataset)
%   pt2Render - scalar (numeric index)
%   append    - (char) text to append to file name
function [] = RenderDatapointNoEdges(path, dat, pt2Render, append)
    arguments
        path(1,:) char, dat(1,1) Dataset, pt2Render(1,1), append(1,:) char
    end
    do_pretty = false;

    t = tic();

    if isempty(dat.img_sz)
        [row,col] = geom.FindCircleCoords(dat.n_nodes);
        row = row .* 22 + 3;
        col = col .* 22 + 3;
        imgSz = [28,28,1];
    else
        [row,col] = PixelRowCol(dat.img_sz);
        imgSz = dat.img_sz;
    end
    
    dpi = 150;
    if do_pretty
        dpi = 300;
    end

    pixelValues = dat.pixels(:,pt2Render);
    if Config.DO_INVERT_COLORS
        pixelValues = ~pixelValues;
    end

    h = figure('Visible', 'off');
    PlotGraph(EDG([]), [], row, col, [], imgSz, pixelValues, false, false, dat.node_name, do_pretty, num2str(pt2Render));
    fig.print(h, path, append, [4,4], dpi);

    Toc(t, toc(t) > 1);
end