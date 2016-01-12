function model =trainPerceptron(trainData,trainLabels,testData,testLabels)

[d,samples]=size(trainData);

class=max(trainLabels);
accuracyTrain=zeros(1,samples*10);
accuracyTest=zeros(1,samples*10);

model=ones(d,class);

for j=1:10
    for i=1:samples
        model=perceptronUpdate(trainData(:,i),model,trainLabels(i,1));
        accuracyTrain((j-1)*samples+i,1) = testPerceptron(model,trainData,trainLabels);
        accuracyTest((j-1)*samples+i,1) = testPerceptron(model,testData,testLabels);
    end

end

f=figure;
set(f,'Name','Perceptron Training','NumberTitle','off');
clf;
xaxis=[1:1:samples*10];
subplot(2,1,1),plot(xaxis,accuracyTrain);
title('traindata');
subplot(2,1,2),plot(xaxis,accuracyTest);
title('testData');


