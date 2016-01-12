function [objective, grad] = nll(score, trainLabels)

% negative log likelihood loss function
% score - n x 1 vector of probability scores in (0, 1)
% trainLabels - n x 1 vector of {-1, +1} binary class labels
% objective - outputs the negative log likelihood of these labels given the
%           probability scores
% grad - n x 1 gradient with respect to the scores of the loss function

objective = - sum(log(score(trainLabels > 0))) - sum(log(1-score(trainLabels < 0)));

grad = zeros(1, length(trainLabels));

grad(trainLabels > 0) = -1 ./ score(trainLabels > 0);
grad(trainLabels < 0) = 1 ./ (1 - score(trainLabels < 0));
