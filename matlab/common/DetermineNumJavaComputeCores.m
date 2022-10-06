% Eli Bowen 1/2017
% RETURNS:
%   n_cores - scalar (int-valued numeric)
function n_cores = DetermineNumJavaComputeCores()
    n_cores = feature('numCores');
    if n_cores == 24
%         if contains(NameOfComputer(), 'boskop')
%             n_cores = 48; % otherwise it's a J node
%         end
    elseif n_cores ~= 16 % 16 is the response of most worker nodes
        if strcmp(NameOfComputer(), 'eb-grangerlab')
            n_cores = 4; % so I can still use the computer
        end
    end
end