function x = sampleMultinomial(probs, n)

% samples n multinomial indices
% prob: k x 1 vector of probabilities (must sum to 1.0)
% n: scalar number of points to sample

cdf = cumsum(probs(:));

x = sum(bsxfun(@lt, rand(n, 1), cdf'), 2);
