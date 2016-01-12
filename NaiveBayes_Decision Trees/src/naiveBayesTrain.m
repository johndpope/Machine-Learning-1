function model = naiveBayesTrain(trainData, trainLabels, params)

% FILL IN YOUR CODE AND COMMENTS HERE

labels= unique(trainLabels);
[l,~]=size(labels);
[d,n]=size(trainData);
index=1;
model=(zeros(d,l));

for i=1:l
    
    
    %find set of all documents whose labels are i 
    set1=find(trainLabels==i);
    [size1,~]=size(set1);
    for j= 1:d
        
        trainDatasub=trainData(:,set1);
        wordcount= sum(trainDatasub(j,:)); 
        wordcount=full(wordcount);
       % display(wordcount);
       % display(size1);
        prob=(wordcount+params)/(size1+2*params);
        proby=size1/n;
        
        probword=prob*proby;
        if probword~=0
            probword=log(probword);
        end
        model(j,i)=probword;
    end
end
    
    %find corresponding probability of each word in Y
    %fi
 % replace this with your actual model
