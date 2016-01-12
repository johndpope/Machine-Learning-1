load ../processedCardio;

nTr = size(trainData, 2);
nTe = size(testData, 2);
numDim = size(trainData, 1);
numClasses = max(trainLabels);
acc=zeros(3,1);

%% Perceptron Experiment
model.W= trainPerceptron(trainData,trainLabels,testData,testLabels);
predict=linearPredict(testData,model);
acc(1,1)=Accuracy(testLabels,predict);

%% Plot results





%% Logistic Regression
[model,predict,maxval]=logRegTest(trainData,trainLabels,testData,testLabels);


%% Plot results

acc(2,1)=maxval;

%% Gaussian Naive Bayes

sigmaVals = 10.^linspace(-2, 1, 100);
alphaVals = 10.^linspace(-2, 1, 5);
accuracy = zeros(length(sigmaVals), length(alphaVals));
bestModel = [];
bestScore = 0;

for i = 1:length(sigmaVals)
    for j = 1:length(alphaVals)
        params.sigma = sigmaVals(i);
        params.alpha = alphaVals(j);
        
        gnbModel = gnbTrain(trainData, trainLabels, params);
        
        gnbLabels = gnbPredict(testData, gnbModel);
        accuracy(i,j) = sum(gnbLabels == testLabels) / length(testLabels);
    
        if accuracy(i,j) > bestScore
            bestScore = accuracy(i,j);
            bestModel = gnbModel;
        end
    end
end
f4=figure;
set(f4,'Name','Parameter Sweep(Gaussian)','NumberTitle','off');
clf;
surf(alphaVals, sigmaVals, accuracy);
xlabel('alpha');
ylabel('sigma');

acc(3,1)=bestScore;

%% Plot Grid Search Results




%% Best Post-hoc Test Scores
f5=figure;
set(f5,'Name','Accuracy','NumberTitle','off');
clf;
bar(acc);
set(gca,'XTickLabel',{'Perceptron', 'Logistic Regression', 'Gaussian NB'});