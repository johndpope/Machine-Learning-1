function [objective, grad] = mlpObjective(model, trainData, trainLabels, lossFunction)

% Computes the value and gradient for weights of a multi-layered perceptron
% model - current model containing cell array W
%   model.W - cell array of weights for each layer
%   model.squashFunction - squashing function applied to each layer
% trainData - d x n matrix of n training data vectors
% trainLabels - n x 1 {-1, 1} binary label vector
% lossFunction - function handle to loss function applied to final output

%% forward propagation
W = model.W;
numLayers = length(W);
[~, scores, hiddenUnits, squashDerivatives] = mlpPredict(trainData, model);

%% back propagation

n = size(trainData, 2);
error = cell(numLayers, 1);

[objective, outputError] = lossFunction(scores, trainLabels);

% Compute the error of the last layer
error{numLayers} = outputError .* squashDerivatives{numLayers};
grad = cell(numLayers, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to compute the gradient for output layer
% This should be about one line of code of the form
% grad{numLayers} = ????
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grad{numLayers}=error{numLayers}*transpose(hiddenUnits{numLayers-1});

% compute gradient for middle layers
for i = numLayers-1:-1:2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to compute the gradient and error for middle layers
% This should be about two lines of code of the form
% error{i} = ????
% grad{i} = ????
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error{i}=(transpose(W{i+1})*error{i+1}).*squashDerivatives{i};
grad{i}=error{i}*transpose(hiddenUnits{i-1});  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to compute the error and gradient for input layer
% This should be about two lines of code of the form
% error{1} = ???
% grad{1} = ???
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error{1}=(transpose(W{2})*error{2}).*squashDerivatives{1};
grad{1}=error{1}*transpose([trainData; ones(1, n)]);

