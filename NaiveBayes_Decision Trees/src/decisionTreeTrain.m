function model = decisionTreeTrain(trainData, trainLabels, params)

% FILL IN YOUR CODE AND COMMENTS HERE


node=fitTree(trainData,0,params,trainLabels);


model = node; % replace this with your actual model
