clear;

close all

rng(0);

load ../synthData.mat;
load ../trueData.mat trueData;

%% run PCA

[newData, variances] = myPca(data);

%% plot variances

figure(1);
stem(variances);
xlabel('Dimension', 'FontSize', 12);
ylabel('Captured Variance', 'FontSize', 12);


figure(2);
plot(newData(1,:), newData(2,:), 'x');
title('Transformed Data After PCA');

%% truncate data

smallData = newData(1:2, :);

%% split data for validation

[d,n] = size(smallData);

% use fraction of data for training

trainInds = rand(n, 1) < 0.5;

trainData = smallData(:, trainInds);
valData = smallData(:, ~trainInds);


%% find Gaussian mixture

numClusters = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14];

valLikelihood = zeros(length(numClusters), 1);

means = cell(length(numClusters), 1);
sigmas = cell(length(numClusters), 1);
clustProbs = cell(length(numClusters), 1);

for k = 1:length(numClusters)
    figure(2 + k);
    [means{k}, sigmas{k}, clustProbs{k}] = gmm(trainData, numClusters(k));
    valLikelihood(k) = gmmLL(valData, means{k}, sigmas{k}, clustProbs{k});
end


%% plot likelihoods

figure(length(numClusters) + 3);

plot(numClusters, valLikelihood);
xlabel('Number of Gaussians', 'FontSize', 12);
ylabel('Log Likelihood of Val. Data');
