
function [trainData,trainLabels,testData,testLabels]= GenerateSyntheticDataL()

trainData= zeros(2,100);
trainLabels=zeros(100,1);
testData=zeros(2,100);
testLabels=zeros(100,1);
for i=1:100
    valueTest=rand(2,1);
    valueTest = valueTest-0.5;
    
    valueTrain= rand(2,1);
    valueTrain=valueTrain-0.5;
    
    if valueTest(1)*valueTest(2)>0
        if valueTest(1)>0
            classTest=1;
        else
            classTest=3;
        end
    else
        if valueTest(1)>0
            classTest=4;
        else
            classTest=2; 
        end
    end
    %%%%%
    if valueTrain(1)*valueTrain(2)>0
        if valueTrain(1)>0
            classTrain=1;
        else
            classTrain=3-ceil(rand(1));
        end
    else
        if valueTrain(1)>0
            classTrain=4;
        else
            classTrain=2+ceil(rand(1)); 
        end
    end
    trainData(:,i)=valueTrain;
    trainLabels(i,1)= classTrain;
    testData(:,i)=valueTest;
    testLabels(i,1)=classTest;
    
end