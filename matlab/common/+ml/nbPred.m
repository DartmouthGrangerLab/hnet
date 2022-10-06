% prediction of naive Bayes classifier with independent Gaussian
% INPUTS
%   model: trained model structure
%   X:     d x n (numeric or logical) - data matrix
% RETURNS
%   y: 1 x n (int-valued numeric) - predicted class label
% written by Mo Chen (sth4nth@gmail.com)
% downloaded by Eli Bowen 12/5/2021 from https://www.mathworks.com/matlabcentral/fileexchange/55864-naive-bayes-classifier
% edited only for argument validation, clarity, and style consistency, then I merged nbGaussPred and nbBernPred.
% Copyright (c) 2016, Mo Chen All rights reserved.
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
function y = nbPred(model, X)
    validateattributes(model, {'struct'}, {'nonempty'}, 1);
    validateattributes(X, {'numeric','logical'}, {'nonempty'}, 2);
    w = model.w;
    mu = model.mu;
    d = size(mu, 1);

    if strcmp(model.dist, 'gauss')
        var = model.var;

        lambda = 1 ./ var;
        ml = mu .* lambda;
        M = bsxfun(@plus, lambda'*X.^2-2*ml'*X, dot(mu, ml, 1)'); % M distance
        c = d*log(2*pi) + 2*sum(log(var), 1)'; % normalization constant
        R = -0.5 .* bsxfun(@plus, M, c);
%         R = exp(R) .* w; % original (R is often too large of a negative number to call exp on)
        R = R + log(w); % should equal log(original), the max of which should be the same index
    elseif strcmp(model.dist, 'bern')
        X = sparse(X);
        R = log(mu+eps)'*X + log(1-mu+eps)'*(1-X); % need eps: log(0) = -Inf, which dominates all other parts of the multiplication
        R = R + log(w);
    elseif strcmp(model.dist, 'multinomial')
        R = mu'*X + w; % calculate the posterior log probability of the samples X
        error('untested performance!');
    else
        error('unexpected model')
    end

    [~,y] = max(R, [], 1);
end