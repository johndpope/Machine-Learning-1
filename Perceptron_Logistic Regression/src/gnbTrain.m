function model = gnbTrain(trainData, trainLabels, params)

% trains a Gaussian naive Bayes model
% trainData: d x n data matrix of n examples
% trainLabels: n x 1 label vector
% params: object containing alpha regularization parameter for Beta prior
%         and sigma parameter for univariate normals
% model: object containing word-conditional and prior log probabilities


[d,samples]=size(trainData);
classes=max(trainLabels);
mu=zeros(d,classes(1,1));
prior=zeros(classes(1,1),1);

for j=1:classes(1,1)
    index=find(trainLabels==j);
    [siz,~]=size(index);
    x_C=trainData(:,index);
    mean= sum(x_C,2)/siz;
    mu(:,j)=mean;
    prior(j,1)=(siz+params.alpha)/(samples+params.alpha*classes(1,1));
end


model.mean=mu;
model.prior=prior;
model.sigma=params.sigma;


