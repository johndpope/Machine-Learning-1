function [nll, gradient] = logRegNLL(W, trainData, trainLabels, params)

% returns the value of the log likelihood for multinomial logistic
% regression and the gradient with respect to the W matrix
% W - d x k matrix of weights
% trainData - d x n matrix of examples as column vectors
% trainLabels - n x 1 matrix of labels (1 through k)
% params - struct containing parameters such as params.lambda
% params.lambda - regularization parameter

%%%%%%%%%%%%%%%%%%%calculate nll
[d,samples]=size(trainData);
[~,classes]=size(W);
mu=zeros(1,samples);
beta=zeros(1,samples);
gradMatrix=zeros(d,classes);



for i=1:samples

Wt=transpose(W);
k= Wt*trainData(:,i);

%% for nll

mu(1,i)=logsumexp(k);
beta(1,i) = Wt(trainLabels(i,1),:)*trainData(:,i);

%% for gradient
y_i=zeros(classes,1);
y_i(trainLabels(i,1),1)=1;
wcX= exp(k);
gamma= wcX/sum(wcX);
gamma=gamma-y_i;

gradMatrix= gradMatrix+trainData(:,i)*transpose(gamma);

end



firstterm=sum(mu);
secondterm=sum(beta);

gradient = params.lambda*W+gradMatrix;

nll= (params.lambda/2*norm(W)^2)+firstterm-secondterm;

















