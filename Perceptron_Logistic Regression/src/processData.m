%%
% Scales and centers the cardiotography data to make it zero-mean unit
% variance
%

load ../Cardiotocography;

means = mean(rawData, 2);

deviations = std(rawData, 1, 2);

% remove uniform features
uniformFeatures = deviations == 0;
rawData(uniformFeatures,:) = [];
means(uniformFeatures) = [];
deviations(uniformFeatures) = [];

[d, n] = size(rawData);

processedData = (rawData - means * ones(1, n)) ./ (deviations * ones (1,n));

rng(1); % use a fixed random seed

nTr = round(0.6 * n);
% nTe = n - nTr;

indices = randperm(n);

trainData = processedData(:, 1:nTr);
trainLabels = rawLabels(1:nTr);

testData = processedData(:, nTr+1:end);
testLabels = rawLabels(nTr+1:end);

save ../processedCardio trainData trainLabels testData testLabels;