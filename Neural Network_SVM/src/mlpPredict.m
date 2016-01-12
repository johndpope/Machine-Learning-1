function [labels, scores, hiddenUnits, squashDerivatives] = mlpPredict(data, model)

% performs forward propagation in multi-layered perceptron
% data - d x n matrix of examples to be classified
% model - struct containing weight cell array and squashing function
% labels - {-1, +1} binary labels for each example in data
% scores - real valued scores output by MLP (e.g., probabilities for logistic squashing)
% hiddenUnits - cell array of hidden unit activations
% squashDerivatives - cell array of derivatives of the squashing function
%           at each layer


%% forward propagation

n = size(data,2);

W = model.W;
squash = model.squashFunction;

numLayers = length(W);

hiddenUnits = cell(numLayers,1);
squashDerivatives = cell(numLayers,1);

[hiddenUnits{1}, squashDerivatives{1}] = squash(W{1}*[data; ones(1, n)]);

for i = 2:numLayers
    [hiddenUnits{i}, squashDerivatives{i}] = squash(W{i}*hiddenUnits{i-1});
end

labels = 2*(hiddenUnits{numLayers}' > 0.5) - 1;

scores = hiddenUnits{numLayers}';


