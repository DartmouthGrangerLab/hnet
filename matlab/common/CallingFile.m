% Eli Bowen 6/10/2021
% doesn't *always* return the name you'd expect, in situations where you're using anonymous functions etc
% RETURNS
%   x - (char) the mfilename() of caller's caller (minus '.m')
%       if the caller's caller is a class method, this is the class name
%       if the caller's caller is a function,     this is the function name
%       if the caller's caller is a script,       this is the script name
function x = CallingFile()
    st = dbstack();
    [~,x,~] = fileparts(st(3).file); % 1 would be CallingFile.m, 2 would be CallingFile()'s caller - I want my caller's caller
end