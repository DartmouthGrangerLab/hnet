% Eli Bowen
% Like matlab's built in strfind(), but it also finds a string in a cell array of strings
% Note matlab's contains(), startsWith(), and endsWith() already support cell arrays of strings, but contains() is much slower than this function
% INPUTS:
%   str      - variable to search within. If a char, returns the result of strfind(). If a cell array of char (e.g. {'hi','there'}), returns indices into the cell array.
%   pattern  - search term (char)
%   is_exact - scalar (logical) if true, will only accept exact string matches. Note search is always case sensitive. Default = 0
% RETURNS:
%   idx - empty array [] if nothing found, otherwise an array of integers
function idx = StringFind(str, pattern, is_exact)
    if iscellstr(str)
        if is_exact
            idx = find(strcmp(str, pattern)); % 20x faster than below
%             idx = find(not(cellfun('isempty', strfind(str, pattern))));
%             exactIndices = [];
%             for i = 1 : numel(idx)
%                 if strcmp(str{idx(i)}, pattern)
%                     exactIndices = [exactIndices, idx(i)];
%                 end
%             end
%             idx = exactIndices;
        else
            idx = find(not(cellfun('isempty', strfind(str, pattern)))); % DON'T USE CONTAINS, as recommended by matlab - it's twice as slow!
%             idx = find(contains(str, pattern, 'IgnoreCase', false)); % slower
        end
    elseif ischar(str)
        if is_exact
            if strcmp(str, pattern)
                idx = 1;
            else
                idx = [];
            end
        else
            idx = strfind(str, pattern);
        end
    else
        error('parameter str is neither string nor cell array of strings!');
    end
end