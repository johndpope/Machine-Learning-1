function [score, models] = crossValidate(trainer, predictor, trainData, trainLabels, folds, params)

% Performs cross validation with random splits
% trainer: function that trains a model from data with the template 
%          model = functionName(trainData, trainLabels, params)
% predictor: function that predicts a label from a single data point
%            label = functionName(data, model)
% trainData: d x n data matrix
% trainLabels: n x 1 label vector
% folds: number of folds to run of validation
% params: auxiliary variables for training algorithm (e.g., regularization
%         parameters)
% score: the accuracy averaged over all folds

scores = zeros(folds, 1);

[~, n] = size(trainData);

indices = 1:n;

% pad indices to make it divide by folds
examplesPerFold = ceil(n / folds);
indices(end+1:examplesPerFold * folds) = nan;
indices = reshape(indices, examplesPerFold, folds);

models = cell(folds, 1);

for i = 1:folds
    trainIndices = indices(:, [(1:(i-1)) (i+1:folds)]);
    trainIndices = trainIndices(~isnan(trainIndices)); % remove padded NaN entries
    testIndices = indices(:, i);
    testIndices = testIndices(~isnan(testIndices)); % remove padded NaN entries

    valData = trainData(:, testIndices);
    valLabels = trainLabels(testIndices);
    
    trData = trainData(:, trainIndices);
    trLabels = trainLabels(trainIndices);
    
    models{i} = trainer(trData, trLabels, params);
    
    predictions = predictor(valData, models{i});
    
    scores(i) = sum(predictions == valLabels) / length(valLabels);
end

score = mean(scores);