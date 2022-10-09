% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% INPUTS
%   path       - (char) output directory
%   classNames - (cell)
%   code
%   bank       - (char) name of component bank to render
%   append     - (char) text to append to file name
function [] = RenderHist(path, classNames, code, bank, append)
    arguments
        path(1,:) char, classNames(:,1) cell, code(1,1) struct, bank char, append(1,:) char
    end

    h = figure('Visible', 'off', 'defaultAxesFontSize', 14); % default = 10
    imagesc(code.hist.(bank));
    colorbar;
    colormap(flipud(redblue()));
    ax = gca();
    try
        ax.CLim = [-max(abs(code.hist.(bank)(:))),max(abs(code.hist.(bank)(:)))];
    end
    ax.YTick = 1:10;
    ax.YTickLabel = classNames;
    ax.XAxis.MinorTick = 'on';
    ax.XRuler.MinorTickValues = 1:10:size(code.hist.(bank), 1);
    xlabel('component num');
    ylabel('class name');
    fig.print(h, path, ['hist_',bank,'_',append], [20,5]);
end