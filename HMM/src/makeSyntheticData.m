rng(0);

n = 1000;

numTypes = 3;
numStates = 4;

d = 2;

% parameter for staying in the same state
sameState = 20;

%% Generate random models

models = cell(numTypes, 1);

for i = 1:numTypes
    models{i}.prior = rand(numStates,1);
    models{i}.prior = models{i}.prior / sum(models{i}.prior);
    
    models{i}.transitions = rand(numStates, numStates) + sameState * eye(numStates);
    models{i}.transitions = models{i}.transitions ./ ...
        (sum(models{i}.transitions, 2) * ones(1, numStates));
    
    models{i}.means = randn(d, numStates);
    models{i}.sigma = cell(numStates, 1);
    for j = 1:numStates
        % generate random PSD matrix
        tmp = randn(d, d);
        models{i}.sigma{j} = tmp * tmp';
    end
end

%% Generate a sequence from each model

trainData = cell(numTypes, 1);
testData = cell(numTypes, 1);
trainStates = cell(numTypes, 1);
testStates = cell(numTypes, 1);

for i = 1:numTypes
    trainData{i} = zeros(d, n);
    testData{i} = zeros(d, n);
 
    trainStates{i} = sampleMarkovChain(models{i}, n);
    testStates{i} = sampleMarkovChain(models{i}, n);

    for j = 1:numStates
        inds = trainStates{i} == j;
        trainData{i}(:, inds) = ...
            sampleGaussian(models{i}.means(:,j), models{i}.sigma{j}, nnz(inds));

        inds = testStates{i} == j;
        testData{i}(:, inds) = ...
            sampleGaussian(models{i}.means(:,j), models{i}.sigma{j}, nnz(inds));
    end
end

%% save files

save ../synthObservations trainData testData;
save ../synthStates trainStates testStates;
save ../trueModels models;

% %% animate training sequences
% figure(1);
% 
% for i = 1:n
%     subplot(311);
%     plot(trainData{1}(1,i), trainData{1}(2,i), 'xr', 'MarkerSize', 10, 'LineWidth', 2);
%     title(sprintf('Model 1. Time step %d, state %d\n', i, trainStates{1}(i)));
%     axis([-5 5 -5 5]);
%     
%     subplot(312);
%     plot(trainData{2}(1,i), trainData{2}(2,i), 'xr', 'MarkerSize', 10, 'LineWidth', 2);
%     title(sprintf('Model 2. Time step %d, state %d\n', i, trainStates{2}(i)));
%     axis([-5 5 -5 5]);
%     
%     subplot(313);
%     plot(trainData{3}(1,i), trainData{3}(2,i), 'xr', 'MarkerSize', 10, 'LineWidth', 2);
%     title(sprintf('Model 3. Time step %d, state %d\n', i, trainStates{3}(i)));
%     axis([-5 5 -5 5]);
% 
%     drawnow;
% end

