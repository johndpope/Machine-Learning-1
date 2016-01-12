
%use naive bayes
folds=10;
score=zeros(10);
for param=1:10
    score(param)=crossvalidate(@naiveBayesTrain,@naiveBayesPredict,trainData,trainLabels,folds,param/10);
end
plot(score);

    
%use decision tree
folds=10;
for param=1:10
    score(param)=crossvalidate(@decisionTreeTrain,@decisionTreePredict,trainData,trainLabels,folds,param);
end
plot(score);  