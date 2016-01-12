function y = logsumexp(X)

% computes log(sum(exp(X))) in numerical stable manner
% X - m x n matrix

[m,n] = size(X);
maxVal = max(X);

y = log(sum(exp(X - ones(m, 1) * maxVal))) + maxVal;
