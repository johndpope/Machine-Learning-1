function [y, grad] = logistic(x)

% Logistic squashing function

y = 1 ./ (1 + exp(-x));

grad = y .* (1-y);


