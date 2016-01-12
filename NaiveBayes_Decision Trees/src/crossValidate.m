function accuracy = crossValidate(trainer, predictor, trainData, trainLabels, folds, params)

% Performs cross validation with random splits
% trainer: function that trains a model from data with the template 
%          model = functionName(trainData, trainLabels, params)
% predictor: function that predicts a label from a single data point
%            label = functionName(data, model)
% trainData: d x n data matrix
% trainLabels: n x 1 label vector
% folds: number of folds to run of validation
% params: auxiliary variables for training algorithm (e.g., regularization
%         parameters)
% score: the accuracy averaged over all folds

% FILL IN YOUR CODE HERE
[~,n]=size(trainData);
indices=crossvalind('KFold',n,folds);
accuracy=0;

for i=1:folds
    index=find(indices~=i);
    subsetX=trainData(:,transpose(index));
    subsetY=trainLabels(transpose(index),:);
    model=trainer(subsetX,subsetY,params);
    
    
    index=find(indices==1);
    testsubsetX=trainData(:,transpose(index));
    testsubsetY=trainLabels(transpose(index),:);
    prediction=predictor(testsubsetX,model);
    
    count=0;
    [size_y,~]=size(testsubsetY);
    for k=1:size_y
        if testsubsetY(k,1)==prediction(k,1)
            count=count+1;
        end
    end
    
    accuracy=accuracy+count/size_y;
    
end
accuracy=accuracy/folds;
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    





