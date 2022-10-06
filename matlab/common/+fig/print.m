% Eli Bowen 10/7/2021
% INPUTS
%   h                  - scalar figure handle or image
%   path               - (char) file path
%   file               - (char) file name
%   paperSz            - OPTIONAL 1 x 2 (numeric) or (char) size of the paper (in inches?) or 'auto' (default = 'auto')
%   dpi                - OPTIONAL scalar (int-valued numeric) resolution (default = 300)
%   do_close_when_done - OPTIONAL scalar (logical) if true, close figure when done printing (default = true)
function [] = print(h, path, file, paperSz, dpi, do_close_when_done)
    validateattributes(h, {'matlab.ui.Figure','numeric','logical'}, {'nonempty'}, 1);
    validateattributes(path, {'char'}, {'nonempty'}, 2);
    validateattributes(file, {'char'}, {'nonempty'}, 3);
    if ~exist('dpi', 'var') || isempty(dpi)
        dpi = 300;
    end
    validateattributes(dpi, {'numeric'}, {'nonempty','scalar','positive','integer'});
    if ~exist('do_close_when_done', 'var') || isempty(do_close_when_done)
        do_close_when_done = true;
    end
    validateattributes(do_close_when_done, 'logical', {'nonempty','scalar'});

    if ~exist(path, 'dir')
        mkdir(path);
    end

    if isa(h, 'matlab.ui.Figure')
        if ~exist('paperSz', 'var') || isempty(paperSz) || strcmp(paperSz, 'auto')
            % determine num subplots
            axes = findall(h, 'type', 'axes');
            posR = zeros(1, numel(axes));
            posC = zeros(1, numel(axes));
            for i = 1 : numel(axes)
                posR(i) = axes(i).Position(2);
                posC(i) = axes(i).Position(1);
            end
            n_rows = numel(unique(posR));
            n_cols = numel(unique(posC));
            paperSz = 5 .* [n_cols,n_rows];
        end
        validateattributes(paperSz, {'numeric'}, {'nonempty','positive','numel',2});

        set(h, 'PaperPosition', [0,0,paperSz(1),paperSz(2)]); % [left, bottom, width, height]

        if endsWith(file, '.png')
            print(h, fullfile(path, file), '-dpng', ['-r',num2str(dpi)]);
        elseif endsWith(file, '.jpg') || endsWith(file, '.jpeg')
            print(h, fullfile(path, file), '-djpeg', ['-r',num2str(dpi)]);
        elseif endsWith(file, '.fig')
            savefig(h, fullfile(path, file), 'compact');
        else % default to png, which is nice and lossless
            print(h, fullfile(path, [file,'.png']), '-dpng', ['-r',num2str(dpi)]);
        end
    else % it's an image
        if islogical(h)
            h = double(h);
        end
        imwrite(h, fullfile(path, file));
        do_close_when_done = false;
    end

    if do_close_when_done
        close(h);
    end
end