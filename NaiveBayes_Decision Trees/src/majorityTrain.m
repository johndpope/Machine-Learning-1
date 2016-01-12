function model = majorityTrain(trainData, trainLabels, params)

% Trains a model that always predicts the most frequent class

allLabels = unique(trainLabels);

counts = zeros(size(allLabels));

for i = 1:length(allLabels)
    counts(i) = sum(trainLabels == allLabels(i));
end

[~, maxIndex] = max(counts);

model.majorityClass = allLabels(maxIndex);
