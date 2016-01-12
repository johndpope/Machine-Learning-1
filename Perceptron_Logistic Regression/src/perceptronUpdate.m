function model = perceptronUpdate(x, model, y)

% updates the weights using the perceptron update rule
% x - single data point
% model - struct containing a weight vector
% y - label index of ground truth
%expects model of dimension dxc and x of dimension dx1

%display(size(model));
%display(size(x));
estimate= transpose(model)*x;
[~,y_predict]=max(estimate);

if(y_predict~=y)
    model(:,y_predict)=model(:,y_predict)-x;
    model(:,y)= model(:,y)+x;
end











