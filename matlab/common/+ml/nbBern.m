% naive bayes classifier with indepenet Bernoulli
% INPUTS
%   X:        d x n (numeric or logical) data matrix
%   labelIdx: 1 x n (int-valued numeric) label {1 .. n_classes}
% RETURNS
%   model: trained model structure
% written by Mo Chen (sth4nth@gmail.com)
% downloaded by Eli Bowen 12/5/2021 from https://www.mathworks.com/matlabcentral/fileexchange/55864-naive-bayes-classifier
% edited only for argument validation, clarity, and style consistency
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
function model = nbBern(X, labelIdx)
    validateattributes(X, {'numeric','logical'}, {'nonempty'}, 1);
    validateattributes(labelIdx, {'numeric'}, {'nonempty','vector'}, 2);
    assert(size(X, 2) == numel(labelIdx));
    labelIdx = labelIdx(:)';
    n = size(X, 2);
    k = max(labelIdx); % n_classes

    E = full(sparse(labelIdx, 1:n, 1, k, n, n)); % create a 1-hot label code (n x k)
    nk = sum(E, 2);         % k x 1 (int-valued numeric) num datapoints per class
    w = nk ./ n;            % k x 1 (numeric)
    dia = diag(1 ./ nk, 0); % k x k (numeric) diagonal matrix (with values of 1./nk along the diagonal)
    mu = X * (E' * dia);
    % above seems 10x faster than below
%     E = sparse(labelIdx, 1:n, 1, k, n, n); % create a 1-hot label code
%     nk = full(sum(E, 2));          % k x 1 (int-valued numeric)
%     w = nk ./ n;                   % k x 1 (numeric)
%     dia = spdiags(1./nk, 0, k, k); % k x k (int-valued numeric) diagonal matrix (with values of 1./nk along the diagonal)
%     R = E' * dia;                  % n x k
%     mu = full(sparse(X) * R);

    model = struct();
    model.name = 'nb';
    model.dist = 'bern';
    model.w = w;   % k x 1
    model.mu = mu; % d x k means
end