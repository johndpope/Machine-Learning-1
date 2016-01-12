function model = logRegTrain(trainData, trainLabels, params, W0)

% trains a multinomial logistic regression by optimizing logRegNLL
% trainData - d x n matrix with each column as an example
% trainLabels - n x 1 vector of ground truth labels
% params - struct containing lambda regularization parameter
% W0 - optional starting point for warm starting optimization

[d,n] = size(trainData);

numClasses = max(trainLabels);

if nargin <= 4
    W0 = zeros(d, numClasses);
end

options = optimoptions(@fminunc, ...
    'DerivativeCheck','off',...
    'GradObj','on',...
    'Display','final',...
    'Algorithm', 'quasi-newton',...
    'MaxIter', 2000);

objective = @(x) logRegNLL(x, trainData, trainLabels, params);
model.W = fminunc(objective, W0, options);

