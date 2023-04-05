% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   label_onehot - n x n_classes (logical)
%   code - n_cmp x n_pts (numeric or logical) component code
% RETURNS
%   hist - n_classes x n_cmp (numeric)
function hist = ClassHistogram(label_onehot, code)
    arguments
        label_onehot(:,:) {mustBeLogical}, code(:,:)
    end
    n_cmp = size(code, 1);
    n_classes = size(label_onehot, 2);
    
    hist = zeros(n_cmp, n_classes); % will be transposed later
    for c = 1 : n_classes
        hist(:,c) = sum(code(:,label_onehot(:,c)), 2);
    end
    hist = hist';
    hist = hist ./ max(abs(hist), [], 1); % normalize to range -1 --> 1
    hist(isnan(hist)) = 0; % divide by zero error
end