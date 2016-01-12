function ll = gaussianLL(data, mean, sigma)

% returns the log likelihood of data points for Gaussian mean and
% covariance sigma
% data: d x n matrix of n data points
% mean: d x 1 mean vector
% sigma: d x d covariance matrix
% ll: n x 1 vector of log likelihoods (sum them to get total log likelihood)

% compute squared distances

[d,n] = size(data);

ll = - (d/2) * log(2*pi) - 0.5 * log(det(sigma));

diff = data - mean * ones(1,n);

ll = ll - 0.5 * sum((sigma \ diff) .* diff);

end