
load ../processedCardio;
%%generate selecting increasing batch sizes of data

[d,n]=size(trainData);
classes=max(trainLabels);
[~,ntest]=size(testData);
acc=zeros(3,int16(n/50)-1);


for i=1:(n/50)
    display(i);
    row_rand=randperm(n);
    trainD = trainData(:,row_rand(:,1:i*50));
    trainL = trainLabels(row_rand(:,1:i*50),:);
    classes=max(trainL);
    
    
%%%%%%%%%%%%%%%%test perceptron
    
    modelP=ones(d,classes);
    for k=1:10
        for j=1:i*50
            modelP=perceptronUpdate(trainD(:,j),modelP,trainL(j,1));
        end
    end
    acc(1,i)= testPerceptron(modelP,testData,testLabels);
    
%%%%%%%%%%%%%%%test Logistic Regression
    
    modelR.W=zeros(d,classes);
    accuLog=0;
    params.lambda=0.01;
    
    modelR=logRegTrain(trainD, trainL, params, modelR.W);
    
    for z=1:ntest
        ktest= transpose(modelR.W)*testData(:,z);
        [~,predicttest]=max(ktest);
        if predicttest==testLabels(z,1)
            accuLog=accuLog+1;
        end
    end
    acc(2,i)= accuLog/ntest;
    
%%%%%%%%%%%%%%%%%%test naiveBayes    
    paramsNB.alpha=0.1;
    paramsNB.sigma=0.1;
   [~,accNB]=gnbTest(trainD,trainL,testData,testLabels,paramsNB);

    acc(3,i)=accNB;
    
end

plot((1:length(acc)).*50,transpose(acc));
legend('Perceptron','Logistic Regression','Gaussian Naive Bayes')

