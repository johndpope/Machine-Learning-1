function [objective, grad] = mlpFlatObjective(x, model, trainData, trainLabels, lossFunction)

% Wrapper to make cell array representation of MLP weights compatible with
% vector-based gradient optimizers

start = 1;
for i = 1:length(model.W)
    model.W{i} = reshape(x(start:start+numel(model.W{i})-1), size(model.W{i},1), size(model.W{i},2));
    start = start+numel(model.W{i});
end

[objective, gradCell] = mlpObjective(model, trainData, trainLabels, lossFunction);

grad = zeros(size(x));
start = 1;
for i = 1:length(model.W)
    grad(start:start+numel(model.W{i})-1) = gradCell{i}(:);
    start = start+numel(model.W{i});
end
