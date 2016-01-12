function Y = linearPredict(data, model)

% predicts the labels of a data set as the class whose weight vector has
% the maximum dot product with the data
% data - d x n matrix of n data points
% model - struct containing model.W, a d x C matrix of weight vectors
[d,samples]=size(data);
[~,classes]=size(model.W);
Y=zeros(samples,1);
for i=1:samples
    estimate= transpose(model.W)*data(:,i);
    [~,y_predict]=max(estimate);
    Y(i,1)=y_predict;
    
end