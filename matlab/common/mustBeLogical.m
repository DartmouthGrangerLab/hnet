% Eli Bowen 11/20/2021
% somehow this isn't built-in...
% INPUTS:
%   x
function [] = mustBeLogical(x)
    try
        assert(islogical(x));
    catch ex
        throwAsCaller(ex);
    end
end