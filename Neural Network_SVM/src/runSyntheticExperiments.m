%%
% Runs multi-layered perceptron, linear SVM, polynomial SVM, and RBF SVM
% on 10 different synthetic data sets, uses cross-validation, and
% evaluates on the test sets.


clear;
close all;

load ../syntheticData;

numDataSets = length(trainData);
numModels = 4;
folds = 5;

testAccuracy = zeros(numDataSets, numModels);


for i = 1:numDataSets
    %% run multi-layered perceptron
    
    structures = {1, 2, 4, [2 2], [2 4], [4 4], [2 2 2]};
    lambdaVals = [0.01, 0.1, 1];
    
    clear params;
    
    params.maxIter = 400;
    params.squashFunction = @logistic;
    params.lossFunction = @nll;
    
    bestParams = [];
    bestScore = 0;
    
    for j = 1:length(structures)
        for k = 1:length(lambdaVals)
            params.numHiddenUnits = structures{j};
            params.lambda = lambdaVals(k);
            
            cvScore = crossValidate(@mlpTrain, @mlpPredict, trainData{i},...
                trainLabels{i}, folds, params);
            
            if cvScore > bestScore
                bestScore = cvScore;
                bestParams = params;
            end
        end
    end
    
    model = mlpTrain(trainData{i}, trainLabels{i}, bestParams);
    predictions = mlpPredict(testData{i}, model);
    testAccuracy(i,1) = sum(predictions == testLabels{i}) / length(testLabels{i});
    
    figure;
    plotSurface(@mlpPredict, model);
    title(sprintf('MLP on Data Set %d', i));
    
    hold on;
    plotData(trainData{i}, trainLabels{i});
    hold off;
    drawnow;
    
    %% run linear SVM
    
    cVals = 10.^linspace(-3, 3, 7);
    
    clear params;
    
    params.kernel = 'linear';
    
    bestParams = [];
    bestScore = 0;
    
    for j = 1:length(cVals)
        params.C = cVals(j);
        cvScore = crossValidate(@kernelSvmTrain, @kernelSvmPredict, ...
            trainData{i}, trainLabels{i}, folds, params);
        
        if cvScore > bestScore
            bestScore = cvScore;
            bestParams = params;
        end
    end
    
    model = kernelSvmTrain(trainData{i}, trainLabels{i}, bestParams);
    predictions = kernelSvmPredict(testData{i}, model);
    testAccuracy(i,2) = sum(predictions == testLabels{i}) / length(testLabels{i});
    
    figure;
    plotSurface(@kernelSvmPredict, model);
    title(sprintf('Linear SVM on Data Set %d', i));
    
    hold on;
    plotData(trainData{i}, trainLabels{i});
    hold off;
    drawnow;
    
    
    %% run polynomial SVM
    
    cVals = 10.^linspace(-3, 3, 7);
    orders = [2 3 4 5 6];
    
    clear params;
    
    params.kernel = 'polynomial';
    
    bestParams = [];
    bestScore = 0;
    
    for j = 1:length(cVals)
        for k = 1:length(orders)
            params.C = cVals(j);
            params.order = orders(k);
            
            cvScore = crossValidate(@kernelSvmTrain, @kernelSvmPredict, ...
                trainData{i}, trainLabels{i}, folds, params);
            
            if cvScore > bestScore
                bestScore = cvScore;
                bestParams = params;
            end
        end
    end
    
    model = kernelSvmTrain(trainData{i}, trainLabels{i}, bestParams);
    predictions = kernelSvmPredict(testData{i}, model);
    testAccuracy(i,3) = sum(predictions == testLabels{i}) / length(testLabels{i});
    
    figure;
    plotSurface(@kernelSvmPredict, model);
    title(sprintf('Polynomial SVM on Data Set %d', i));
    
    hold on;
    plotData(trainData{i}, trainLabels{i});
    hold off;
    drawnow;
    
    
    %% run RBF SVM
    
    cVals = 10.^linspace(-3, 3, 7);
    sigmaVals = 10.^linspace(-2, 1, 5);
    
    clear params;
    
    params.kernel = 'rbf';
    
    bestParams = [];
    bestScore = 0;
    
    for j = 1:length(cVals)
        for k = 1:length(sigmaVals)
            params.C = cVals(j);
            params.sigma = sigmaVals(k);
            
            cvScore = crossValidate(@kernelSvmTrain, @kernelSvmPredict, ...
                trainData{i}, trainLabels{i}, folds, params);
            
            if cvScore > bestScore
                bestScore = cvScore;
                bestParams = params;
            end
        end
    end
    
    model = kernelSvmTrain(trainData{i}, trainLabels{i}, bestParams);
    predictions = kernelSvmPredict(testData{i}, model);
    testAccuracy(i,4) = sum(predictions == testLabels{i}) / length(testLabels{i});
    
    figure;
    plotSurface(@kernelSvmPredict, model);
    title(sprintf('RBF SVM on Data Set %d', i));
    
    hold on;
    plotData(trainData{i}, trainLabels{i});
    hold off;
    drawnow;
    
end

%% Print accuracy results
fprintf('\n---------------------------------------\nTest Accuracy\n---------------------------------------\n');
fprintf('Set\tMLP\tLinear\tPoly\tRBF\n');
for i = 1:numDataSets
    fprintf('%d', i);
    maxScore = max(testAccuracy(i,:));
    for j = 1:numModels
        fprintf('\t%2.3f', testAccuracy(i,j));
        if maxScore == testAccuracy(i,j)
            fprintf('*');
        end
    end
    fprintf('\n');
end
fprintf('* best scoring method per data set\n');

