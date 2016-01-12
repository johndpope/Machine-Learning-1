function accuracy= testPerceptron(model,testData,testLabels)

[~,samples]=size(testData);
count=0;

for i=1:samples
    estimate= transpose(model)*testData(:,i);
    [~,y_predict]=max(estimate);
    
    if y_predict==testLabels(i,1)
        count= count+1;
    end
    
end 
accuracy=count/samples;
    
    
    
    
    
    
    
    