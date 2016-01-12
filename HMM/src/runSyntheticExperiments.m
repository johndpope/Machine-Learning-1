clear;
rng(1);

load ../synthObservations.mat
load ../trueModels.mat
load ../synthStates.mat

%% try latent state inference with true model

pState = cell(3,3);
logLikelihood = zeros(3,3);


fprintf('Log Likelihood of Data Under True Models\n');
fprintf('True Model\t1\t\t2\t\t3\n')

for dataType = 1:3
    fprintf('Data %d\t', dataType);
    for modelType = 1:3
        [pState{dataType, modelType}, logLikelihood(dataType, modelType)] = ...
            myHmmInferStates(trainData{dataType}, models{modelType});
        fprintf('%2.2d\t', logLikelihood(dataType, modelType));
    end
    fprintf('\n');
end

%% learn models

figure(1);

learnedModels = cell(3,1);

params.numStates = 4;
params.alpha = 1e-3;
params.maxIter = 1000;
params.draw = true;

for dataType = 1:3
    learnedModels{dataType} = myHmmTrain(trainData{dataType}, params);
end

%% test models

fprintf('Training Data Log Likelihood\n');
fprintf('Learned Model\t1\t\t2\t\t3\n')

for dataType = 1:3
    fprintf('Data %d\t', dataType);
    for modelType = 1:3
        [pState{dataType, modelType}, logLikelihood(dataType, modelType)] = ...
            myHmmInferStates(trainData{dataType}, learnedModels{modelType});
        fprintf('%2.2d\t', logLikelihood(dataType, modelType));
    end
    fprintf('\n');
end


fprintf('\nHeld-out Log Likelihood\n');
fprintf('Learned Model\t1\t\t2\t\t3\n')

for dataType = 1:3
    fprintf('Data %d\t', dataType);
    for modelType = 1:3
        [pState{dataType, modelType}, logLikelihood(dataType, modelType)] = ...
            myHmmInferStates(testData{dataType}, learnedModels{modelType});
        fprintf('%2.2d\t', logLikelihood(dataType, modelType));
    end
    fprintf('\n');
end

%% check that correct models have highest likelihood

for dataType = 1:3
    [~, i] = max(logLikelihood(dataType, :));
    fprintf('Best HMM for test data %d was %d\n', dataType, i);
    if i ~= dataType
        fprintf('Something may be wrong. The best HMM for this data was\n   not trained on data from the same distribution.\n');
    end
end

%% plot learned Gaussians

figure(2);
clf;
for type = 1:3
    subplot(2, 3, type);
    
    plot(trainData{type}(1,:), trainData{type}(2,:), '.');
    hold on;
    plotGMM(models{type}.means, models{type}.sigma);
    hold off;
    title(sprintf('True Model %d', type));
    
    subplot(2, 3, type+3);
    
    plot(trainData{type}(1,:), trainData{type}(2,:), '.');
    hold on;
    plotGMM(learnedModels{type}.means, learnedModels{type}.sigma);
    hold off;
    title(sprintf('Learned Model %d', type));
end

