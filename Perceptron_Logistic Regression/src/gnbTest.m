function [trainAcc,testAcc]= gnbTest(trainData,trainLabels,testData,testLabels,params)

model=gnbTrain(trainData,trainLabels,params);
predict=gnbPredict(testData,model);

[samples,~]=size(predict);
accuracy=0;
for i =1:samples;
    if predict(i,1)==testLabels(i,1);
        accuracy=accuracy+1;
    end
end

testAcc=accuracy/samples;


predict=gnbPredict(trainData,model);

[samples,~]=size(predict);
accuracy=0;
for i =1:samples;
    if predict(i,1)==trainLabels(i,1);
        accuracy=accuracy+1;
    end
end

trainAcc=accuracy/samples;

end