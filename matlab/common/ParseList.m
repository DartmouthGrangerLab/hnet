% Eli Bowen 2/15/2022
% INPUTS
%   str - list, either as a cell array of chars or as comma delimited text
% RETURNS
%   str - cell array of chars
function str = ParseList(str)
    validateattributes(str, {'cell','char','string'}, {}, 1);
    if isempty(str)
        str = {};
        return
    end

    if isstring(str)
        str = char(str);
    end

    if ischar(str)
        str = strsplit(str, ',');
    elseif iscell(str)
        assert(all(cellfun(@ischar, str))); % nothing to do
    end
end