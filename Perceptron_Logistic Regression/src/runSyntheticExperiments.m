%% Create synthetic data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 n = 100;
 d = 2;
 numClasses = 4;
% 
% % rng(1);
% 
 trainData = randn(d, n);
 testData = randn(d, n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vals = linspace(-1, 1, sqrt(n));
% count = 1;
% for i = 1:length(vals)
%     for j = 1:length(vals)
%         trainData(:, count) = [vals(i); vals(j)];
%         count = count + 1;
%     end
% end

%%Accuracy matrix
acc=zeros(3,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 model.W = randn(d, numClasses);
% 
 trainLabels = linearPredict(trainData, model);
 testLabels = linearPredict(testData, model);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[trainData,trainLabels,testData,testLabels]=GenerateSyntheticDataNL;

f1=figure;
set(f1,'Name','Linear Data','NumberTitle','off');
clf;
plotPredictions(trainData, trainLabels, trainLabels);

%% Perceptron Experiment

model.W= trainPerceptron(trainData,trainLabels,testData,testLabels);
predict=linearPredict(testData,model);
acc(1,1)=Accuracy(testLabels,predict);


%% Plot results
f2=figure;
set(f2,'Name','PerceptronResults','NumberTitle','off');
clf;
plotPredictions(testData, testLabels, predict);




%% Logistic Regression
[model,predict,maxval]=logRegTest(trainData,trainLabels,testData,testLabels);




%% Plot results
f3=figure;
set(f3,'Name','LogRegression','NumberTitle','off');
clf;
plotPredictions(testData,testLabels,predict);

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
f5=figure;
set(f5,'Name','Gaussian Bayes','NumberTitle','off');
clf;
gnbLabels = gnbPredict(testData, bestModel);
plotPredictions(testData, testLabels, gnbLabels);
acc(3,1)=Accuracy(testLabels,gnbLabels);

%% Plot Grid Search Results




%% Best Post-hoc Test Scores
f5=figure;
set(f5,'Name','Accuracy','NumberTitle','off');
clf;
bar(acc);
set(gca,'XTickLabel',{'Perceptron', 'Logistic Regression', 'Gaussian NB'});


