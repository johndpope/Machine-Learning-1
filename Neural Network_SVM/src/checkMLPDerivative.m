%%
% Script to initialize a random weight MLP and check the derivative of
% the loss function
%

clear;

load ../syntheticData;

rng(0);

params.numHiddenUnits = [2, 3];
params.maxIter = 1000;
params.squashFunction = @logistic;
params.lossFunction = @nll;
params.lambda = 1;


numHiddenUnits = params.numHiddenUnits;
numLayers = length(numHiddenUnits) + 1;

trainData{1} = trainData{1}(:,1);
trainLabels{1} = trainLabels{1}(1);

[d,n] = size(trainData{1});

% initialize W randomly
model.W = cell(numLayers, 1);
model.W{1} = randn(numHiddenUnits(1), d + 1);
for i = 2:numLayers-1
    model.W{i} = randn(numHiddenUnits(i), numHiddenUnits(i-1));
end
model.W{numLayers} = randn(1, numHiddenUnits(end));

model.squashFunction = params.squashFunction;

W0 = [];
start = 1;
for i = 1:length(model.W)
    W0(start:start+numel(model.W{i})-1) = model.W{i}(:);
    start = start+numel(model.W{i});
end

options = optimoptions(@fminunc, ...
    'DerivativeCheck','on',... 
    'GradObj','on',...
    'Display','final',...
    'Algorithm', 'quasi-newton',...
    'Diagnostics', 'on',...
    'FinDiffType', 'forward',...
    'MaxIter', 20);

objective = @(x) mlpFlatObjective(x, model, trainData{1}, trainLabels{1}, @nll);
x = fminunc(objective, W0, options);
%% check derivative of loss function

objective = @(x) nll(x, trainLabels{i});
x0 = rand(size(trainLabels{i}));
x = fminunc(objective, x0, options);

%% check derivative of logistic squashing function

x0 = randn;
x = fminunc(@logistic, x0, options);



