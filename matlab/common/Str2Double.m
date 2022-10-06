% like matlab's str2double(), but also supports mathematical equations (e.g. '3/4') and special keywords:
%   'half' --> 0.5
%   'quarter', 'qtr' = 0.25
%   'tenth' = 0.1
% INPUTS
%   x
% RETURNS
%   x
% see also str2double
function x = Str2Double(x)
    if strcmp(x, 'half')
        x = 0.5;
    elseif strcmp(x, 'quarter') || strcmp(x, 'qtr')
        x = 0.25;
    elseif strcmp(x, 'tenth')
        x = 0.1;
    elseif ischar(x) || isstring(x)
        x = eval(x);
    else
        x = str2double(x);
    end
end