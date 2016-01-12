
loadAllData;

%% Filter by information gain

d = 5000;

gain = calculateInformationGain(trainData, trainLabels);
[~, I] = sort(gain, 'descend');

trainData = trainData(I(1:d), :);
testData = testData(I(1:d), :);

%% Try naive Bayes with hard-coded alpha value

nbParams.alpha = 1;

nbModel = naiveBayesTrain(trainData, trainLabels, nbParams);

% compute training accuracy

nbTrainPredictions = zeros(length(trainLabels), 1);

for i = 1:length(trainLabels)
    nbTrainPredictions(i) = naiveBayesPredict(trainData(:,i), nbModel);
end

nbTrainAccuracy = nnz(nbTrainPredictions == trainLabels) ./ length(trainLabels)

% compute testing accuracy

nbPredictions = zeros(length(testLabels), 1);

for i = 1:length(testLabels)
    nbPredictions(i) = naiveBayesPredict(testData(:,i), nbModel);
end

nbAccuracy = nnz(nbPredictions == testLabels) ./ length(testLabels)


%% Try decision tree with hard-coded maximum depth

dtParams.maxDepth = 16;

dtModel = decisionTreeTrain(trainData, trainLabels, dtParams);

dtPredictions = zeros(length(testLabels), 1);

% compute training accuracy

dtTrainPredictions = zeros(length(trainLabels), 1);

for i = 1:length(trainLabels)
    dtTrainPredictions(i) = decisionTreePredict(trainData(:,i), dtModel);
end

dtTrainAccuracy = nnz(dtTrainPredictions == trainLabels) ./ length(trainLabels)

% compute testing accuracy

for i = 1:length(testLabels)
    dtPredictions(i) = decisionTreePredict(testData(:,i), dtModel);
end

dtAccuracy = nnz(dtPredictions == testLabels) ./ length(testLabels)


