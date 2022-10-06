% Eli Bowen 12/17/2021
% less error-prone version of matlab's plot()
% matlab's plot is annoying about x and y looking just right - this function pleases it
% also supports:
%   an additional name-value argument "alpha", for color transparency
%   'Color' = 'redblue' (gradient red to blue)
% INPUTS
%   x
%   y
%   other arguments to matlab's plot()
% RETURNS
%   h - plot handle
% see also plot
function h = plot(x, y, varargin)
    validateattributes(x, {'numeric','logical'}, {'nonempty'}, 1);
    validateattributes(y, {'numeric','logical'}, {'nonempty'}, 2);
    x = double(x);
    y = double(y);

    idx = find(strcmpi(varargin, 'alpha'));
    if ~isempty(idx)
        alpha = varargin{idx+1};
        validateattributes(alpha, {'numeric','logical'}, {'nonempty','nonnegative'});
        assert(all(alpha <= 1));
        varargin([idx,idx+1]) = [];
        
        if isscalar(alpha) && alpha == 0
            return
        elseif isscalar(alpha) && alpha == 1
            h = fig.plot(x, y, varargin); % just call me again now that we've removed this param
            return
        elseif ~isscalar(alpha) && islogical(alpha) || all(alpha == 0 | alpha == 1)
            assert(numel(alpha) == size(x, 2)); % "The plot function plots columns of Y versus columns of X."
            h = fig.plot(x(:,logical(alpha)), y(:,logical(alpha)), varargin{:});
            return
        end
    end

    is_x_ndvec  = (sum(size(x) ~= 1) == 1);
    is_y_ndvec  = (sum(size(y) ~= 1) == 1);
    is_x_scalar = (sum(size(x) ~= 1) == 0);
    is_y_scalar = (sum(size(y) ~= 1) == 0);
    if (~is_x_ndvec && ~is_x_scalar) || (~is_y_ndvec && ~is_y_scalar) || (is_x_scalar && is_y_scalar) % if either is an nd matrix, or both are scalars
        x = squeeze(x);
        y = squeeze(y);
    elseif is_x_scalar && is_y_ndvec
        x = repmat(x, 1, numel(y));
        y = squeeze(y)';
    elseif is_x_ndvec && is_y_scalar
        x = squeeze(x)';
        y = repmat(y, 1, numel(x));
    elseif is_x_ndvec && is_y_ndvec
        x = squeeze(x)';
        y = squeeze(y)';
    else
        error('bug in code - all conditions should be handled');
    end

    idx = find(strcmpi(varargin, 'color'));
    if ~isempty(idx) && strcmpi(varargin{idx+1}, 'redblue')
        varargin([idx,idx+1]) = [];
        
        if isvector(x)
            n_pts_in_line = numel(x);
            h = plot(x, y, varargin{:});
            drawnow % for some reason this must be right here
            cmap = redblue(n_pts_in_line);
            cmap = uint8(255 .* cat(2, cmap, ones(n_pts_in_line, 1))');
            set(h.Edge, 'ColorBinding', 'interpolated', 'ColorData', cmap); % undocumented matlab
        else
            hold on
            for i = 1 : size(x, 2)
                n_pts_in_line = size(x, 1);
                h = plot(x(:,i), y(:,i), varargin{:});
                drawnow % for some reason this must be right here
                cmap = redblue(n_pts_in_line);
                cmap = uint8(255 .* cat(2, cmap, ones(n_pts_in_line, 1))');
                set(h.Edge, 'ColorBinding', 'interpolated', 'ColorData', cmap); % undocumented matlab
            end
        end
    else
        h = plot(x, y, varargin{:}); % pass through to matlab
    end

    if exist('alpha', 'var') && ~isempty(alpha) && ~all(alpha == 1)
        if isscalar(alpha)
            alpha = repmat(alpha, numel(h), 1);
        end
        for i = 1 : numel(h) % for each line we've plotted
            h(i).Color(4) = double(alpha(i)); % undocumented matlab
        end
    end
end