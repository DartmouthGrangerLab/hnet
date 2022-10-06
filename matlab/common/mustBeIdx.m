% Eli Bowen 11/19/2021
% count data that can't be 0 (aka positive integers)
% INPUTS:
%   x
function [] = mustBeIdx(x)
    try
        mustBeNumeric(x);
        mustBePositive(x);
        mustBeInteger(x);
        mustBeNonNan(x);
    catch ex
        throwAsCaller(ex);
    end
end