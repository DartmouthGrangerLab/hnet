% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   data           - n_nodes x n (logical) node activations
%   didx           - n_edges x 2 (numeric index)
%   edgeTypeFilter - ? x 1 (EDG enum)
% RETURNS
%   relations - n_edges x n (EDG enum)
function edgeStates = GetEdgeStates(data, didx, edgeTypeFilter)
    arguments
        data(:,:) {mustBeLogical}, didx(:,2) {mustBeIdx}, edgeTypeFilter(:,1) EDG
    end
    n = size(data, 2);
    n_edges = size(didx, 1);
    
    temp = data(didx(:,1),:).*2 + data(didx(:,2),:);
    edgeStates = EDG(zeros(n_edges, n, "uint8"));
    edgeStates(temp == 0) = EDG.NOR;
    edgeStates(temp == 1) = EDG.NCONV;
    edgeStates(temp == 2) = EDG.NIMPL;
    edgeStates(temp == 3) = EDG.AND;
    
    edgeStates = FilterEdgeType(edgeStates, edgeTypeFilter);
end