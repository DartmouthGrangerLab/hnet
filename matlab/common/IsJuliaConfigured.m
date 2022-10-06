% Eli Bowen 5/2022
% checks whether Julia is installed
% RETURNS
%   tf:      scalar (logical)
%   version: (char) Julia version information
function [tf,version] = IsJuliaConfigured()
    [status,cmdout] = system('julia --version');
    tf = (status == 0); % 0 = no error
    
    version = '';
    if tf
        version = strrep(cmdout, 'julia version ', '');
    end
end