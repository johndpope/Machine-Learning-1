function y = logsumexp(X, dim)

% computes log(sum(exp(X))) in numerical stable manner
% X - m x n matrix

if nargin == 1
    dim = 1;
end

[m,n] = size(X);
maxVal = max(X, [], dim);

y = log(sum(exp(X - ones(m, 1) * maxVal), dim)) + maxVal;
