% Eli Bowen 3/2022
% INPUTS
%   hist - n_classes x n_discriminators (non-negative numeric)
% RETURNS
%   discriminability - 1 x n_discriminators (numeric ranged 0 --> 1, can be nan)
%   sharedness - 1 x n_discriminators (numeric, can be nan) the fraction of classes that a discriminator is selective for
function [discriminability,sharedness] = SparseClassResponderDiscriminability(hist)
    validateattributes(hist, {'numeric'}, {'nonnegative'}, 1);
    n_classes = size(hist, 1);

    hist = hist ./ max(hist, [], 1); % must be ranged 0 --> 1 since the sigmoid ranges 0 --> 1
    
    % if all bars of a discriminator's hist are > 0.5, a discriminator is more invariant than variant to it
    % this function will return NaN in such cases, but often you can set those nans to 0 (no discriminability)

    % fit a logistic / sigmoid
%     func = @(b,x)(1 ./ (1 + exp(-b(1)*(x-b(2)))));
%     b0 = [1,n_classes/2]; % initial values fed into the iterative algorithm
%     for i = 1 : n_discriminators
%         bFit = nlinfit(1:n_classes, sort(hist(:,i)), func, b0);
%         % b(1) = curve steepness
%         % b(2) = midpoint x location
%         discriminability(i) = bFit(1);
%         sharedness(i) = n_classes - bFit(2);
%     end

    % a simpler way
    mask = (hist > 0.5); % nans will be false
    nHigh = sum(mask, 1);
    histHigh = hist;
    histLow = hist;
    histHigh(~mask) = 0;
    histLow(mask) = 0;
    discriminability = sum(histHigh, 1) ./ nHigh - sum(histLow, 1) ./ (n_classes - nHigh);
    % above is same speed as below on cpu, 100x faster on gpu, verified identical
%     discriminability = zeros(1, n_discriminators, 'like', hist);
%     for i = 1 : n_discriminators
%         discriminability(i) = mean(hist(mask(:,i),i)) - mean(hist(~mask(:,i),i));
%     end

    discriminability = (discriminability - 0.5) .* 2; % was ranged 0.5 --> 1, now 0 --> 1

    if nargout() > 1
        sharedness = sum(mask, 1) ./ n_classes; % 1 x n_discriminators
        sharedness(isnan(discriminability)) = NaN; % if one is nan, both should be
    end

    %TODO: deal with negative histograms
    %PROBLEM: what does it mean to be <0?
    %   also, if we leave things 0-->1, we can simply use the dilated image as a feature, because we'll be able to use ORs (an unused synapse won't count against us)
end