% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% filter edges, setting all to n/a except those listed in edgeTypeFilter
% INPUTS
%   edgeStates
%   edgeTypeFilter
% RETURNS
%   edgeStates
function edgeStates = FilterEdgeType(edgeStates, edgeTypeFilter)
    arguments
        edgeStates(:,:) EDG, edgeTypeFilter(:,1) EDG
    end

    % e.g. with fully connected, there are SOOO many NCONV and NIMPL edges, so we just do AND
    if ~isempty(edgeTypeFilter)
        mask = false(size(edgeStates));
        for i = 1 : numel(edgeTypeFilter)
            mask = mask | edgeStates == edgeTypeFilter(i);
        end
        edgeStates(~mask) = EDG.NULL; % with fully connected, there are SOOO many NCONV and NIMPL edges
    end
end