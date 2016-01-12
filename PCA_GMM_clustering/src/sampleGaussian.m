function x = sampleGaussian(mu, sigma, n)

% Samples from a multivariate Gaussian with mean mu and covariance matrix
% sigma
% mu: d x 1 vector
% sigma: d x d covariance matrix
% n: scalar count of points to sample
% x = d x n matrix of sample (in columns)

d = length(mu);

A = chol(sigma);

x = A * randn(d, n) + mu * ones(1, n);

