function x = sampleMultinomial(probs, n)

% samples n multinomial indices
% prob: k x 1 vector of probabilities (must sum to 1.0)
% n: scalar number of points to sample

cdf = cumsum(probs(:));

x = sum(bsxfun(@gt, rand(n, 1), cdf'), 2) + 1;
x(x > length(probs)) = length(probs);
