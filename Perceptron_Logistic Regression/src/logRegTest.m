%% test log regression
function [maxModel,y_out,maxval] = logRegTest(trainData,trainLabels,testData,testLabels)
[d,samples]=size(trainData);
[~,samplestest]=size(testData);

classes=max(trainLabels);

testAcc=zeros(1,10);
trainAcc=zeros(1,10);
model.W=ones(d,classes);
maxModel=0;
maxval=0;
predictt=zeros(samplestest,1);
for j=1:1:10;
    params.lambda=j/100;
    model= logRegTrain(trainData,trainLabels,params,model.W);

    accuracyTrain=0;
    accuracyTest=0;
Wt=transpose(model.W);

for i=1:samples
    
    k= Wt*trainData(:,i);
    [~,predict]=max(k);
   
    
    if predict==trainLabels(i,1)
        accuracyTrain=accuracyTrain+1;
    end
      
end


for z=1:samplestest
    ktest= Wt*testData(:,z);
    [~,predicttest]=max(ktest);
    predictt(z,1)=predicttest;
    if predicttest==testLabels(z,1)
        accuracyTest=accuracyTest+1;
    end
end

    testAcc(1,j)=accuracyTest/samples;
    if maxval<testAcc(1,j)
        maxModel=model;
        maxval=testAcc(1,j);
        lambda=params.lambda;
        y_out=predictt;
    end
    trainAcc(1,j)=accuracyTrain/samples;
    
    
end   
fk=figure;
set(fk,'Name','Regression accuracy wrt lambda','NumberTitle','off');
clf;
xaxis=[0.01:0.01:0.1];
subplot(2,1,1),plot(xaxis,trainAcc);
title('traindata');
subplot(2,1,2),plot(xaxis,testAcc);
title('testData');
        
end












