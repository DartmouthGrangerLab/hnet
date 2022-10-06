% Eli Bowen 4/2022
% INPUTS:
%   t       - scalar (numeric) value returned by tic()
%   verbose - OPTIONAL scalar (logical) default = true
%   append  - OPTIONAL (char)
function [] = Toc(t, verbose, append)
    t = toc(t);

    if ~exist('verbose', 'var') || isempty(verbose) || verbose == true
        if t > 60 * 60 * 2 % > 2 hours
            txt = [num2str(t, '%.0f'),' s (',num2str(t / 60 / 60),' hr)'];
        else
            txt = [num2str(t, '%.4f'),' s'];
        end

        if exist('append', 'var') && ~isempty(append)
            disp([CallingFile(),' (',append,') took ',txt]);
        else
            disp([CallingFile(),' took ',txt]);
        end
    end
end