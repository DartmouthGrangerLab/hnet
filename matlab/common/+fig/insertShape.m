% Eli Bowen 5/2022
% like matlab's built-in insertShape(), but:
%   for shape = 'line', supports:
%       faster render times
%       color = 'redblue'
%       alpha parameter for transparency (TODO)
% INPUTS
%   img
%   shape - (char)
%   position
%   varargin
% RETURNS
%   img - n_rows x n_cols x 3
% see also insertShape
function img = insertShape(img, shape, position, varargin)
    idx = find(strcmpi(varargin, 'alpha'));
    if ~isempty(idx)
        alpha = varargin{idx+1};
        validateattributes(alpha, {'numeric','logical'}, {'nonempty','nonnegative'});
        assert(all(alpha <= 1));
        varargin([idx,idx+1]) = [];
        
        if isscalar(alpha) && alpha == 0
            return % do nothing
        elseif isscalar(alpha) && alpha == 1
            img = fig.insertShape(img, shape, position, varargin{:}); % just call me again now that we've removed this param
            return
        elseif ~isscalar(alpha) && islogical(alpha) || all(alpha == 0 | alpha == 1)
            assert(numel(alpha) == size(position, 1));
            img = fig.insertShape(img, shape, position(logical(alpha),:), varargin{:}); % recurse
            return
        end
    end
    
    do_smooth_edges = false;
    idx = find(strcmpi(varargin, 'SmoothEdges'));
    if ~isempty(idx)
        do_smooth_edges = varargin{idx+1};
    end
    
    color = 'yellow'; % matlab's ugly default
    idx = find(strcmpi(varargin, 'color'));
    if ~isempty(idx)
        color = varargin{idx+1};
    end
    
    lineWidth = 1; % matlab's default
    idx = find(strcmpi(varargin, 'LineWidth'));
    if ~isempty(idx)
        lineWidth = varargin{idx+1};
    end

    if strcmpi(shape, 'line')
        n_lines = size(position, 1);
        
        idx = find(strcmpi(varargin, 'color'));
        if strcmp(color, 'redblue')
            n_color_stops = 10;
            cmap = redblue(n_color_stops+1);
            cmap(n_color_stops/2 + 1,:) = []; % excluding the white middle part
            temp = zeros(n_lines * n_color_stops, 4); % n_lines*n_color_stops x 4 (numeric)
            color = zeros(n_lines * n_color_stops, 3); % n_lines*n_color_stops x 3 (numeric)
            for i = 1 : n_color_stops
                temp((i-1)*n_lines + (1:n_lines),1) = position(:,1) + (position(:,3) - position(:,1)) .* ((i-1) / n_color_stops);
                temp((i-1)*n_lines + (1:n_lines),2) = position(:,2) + (position(:,4) - position(:,2)) .* ((i-1) / n_color_stops);
                temp((i-1)*n_lines + (1:n_lines),3) = position(:,1) + (position(:,3) - position(:,1)) .* (i / n_color_stops);
                temp((i-1)*n_lines + (1:n_lines),4) = position(:,2) + (position(:,4) - position(:,2)) .* (i / n_color_stops);
                color((i-1)*n_lines + (1:n_lines),1) = cmap(i,1);
                color((i-1)*n_lines + (1:n_lines),2) = cmap(i,2);
                color((i-1)*n_lines + (1:n_lines),3) = cmap(i,3); % ugh yes matlab can't handle this in one line
            end
            
            varargin([idx,idx+1]) = [];
            img = fig.insertShape(img, shape, temp, 'Color', color, varargin{:}); % recurse without 'redblue'
        elseif do_smooth_edges || lineWidth > 3
            img = insertShape(img, shape, position, varargin{:});
        else
            % matlab's insertShape doesn't support opacity (but also this code is faster)
            if strcmp(color, 'yellow')
                color = [1,1,0];
            elseif strcmp(color, 'blue')
                color = [0,0,1];
            elseif strcmp(color, 'green')
                color = [0,1,0];
            elseif strcmp(color, 'red')
                color = [1,0,0];
            elseif strcmp(color, 'cyan')
                color = [0,1,1];
            elseif strcmp(color, 'magenta')
                color = [1,0,1];
            elseif strcmp(color, 'black')
                color = [0,0,0];
            elseif strcmp(color, 'white')
                color = [1,1,1];
            end
            if isvector(color)
                color = repmat(color(:)', n_lines, 1);
            end
            if ~exist('alpha', 'var') || isempty(alpha)
                alpha = 1;
            end
            if isscalar(alpha)
                alpha = repmat(alpha, n_lines, 1);
            end
            for i = 1 : n_lines
                pix = DrawSkinnyLineSqr(position(i,1:2), position(i,3:4), 1);
                if lineWidth == 2
                    pix = cat(1, pix, [pix(:,1)+1,pix(:,2)], [pix(:,1),pix(:,2)+1]);
                elseif lineWidth == 3
                    pix = cat(1, pix, [pix(:,1)+1,pix(:,2)], [pix(:,1),pix(:,2)+1], [pix(:,1)-1,pix(:,2)], [pix(:,1),pix(:,2)-1]);
                end
                pix(pix(:,1) < 1 | pix(:,2) < 1 | pix(:,1) > size(img, 2) | pix(:,2) > size(img, 1),:) = [];
                if alpha(i) == 1
                    for j = 1 : size(pix, 1)
                        img(pix(j,2),pix(j,1),:) = color(i,:);
                    end
                else
                    for j = 1 : size(pix, 1)
                        img(pix(j,2),pix(j,1),1) = wmean([img(pix(j,2),pix(j,1),1),color(i,1)], [1-alpha(i),alpha(i)]); % use temp as alpha
                        img(pix(j,2),pix(j,1),2) = wmean([img(pix(j,2),pix(j,1),2),color(i,2)], [1-alpha(i),alpha(i)]); % use temp as alpha
                        img(pix(j,2),pix(j,1),3) = wmean([img(pix(j,2),pix(j,1),3),color(i,3)], [1-alpha(i),alpha(i)]); % use temp as alpha
                    end
                end
            end
        end
    else
        assert(all(alpha == 1), 'alpha only supported for shape == line');
        img = insertShape(img, shape, position, varargin{:});
    end
end