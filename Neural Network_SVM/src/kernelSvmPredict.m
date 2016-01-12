function [labels, score] = kernelSvmPredict(data, model)

% predicts the labels from a kernel SVM using one of the three preset
% kernel types
% data - d x n matrix of n examples
% model - struct containing the support vectors, their alpha values and
%        labels, the bias term and kernel parameters
% labels - returned vector of n {-1,1} binary class labels
% score - returned vector of n real valued prediction scores w
%

params = model.params; % for convenience

% Compute the gram matrix for the selected kernel
if isfield(params, 'kernel') && strcmp(params.kernel, 'rbf')
    gramMatrix = rbfKernel(data, model.sv, params.sigma);
elseif isfield(params, 'kernel') && strcmp(params.kernel, 'polynomial')
    gramMatrix = polynomialKernel(data, model.sv, params.order);
else
    % use a linear kernel by default
    gramMatrix = linearKernel(data, model.sv);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code below to compute the prediction score given the Gram
% matrix, model.alphas, model.svLabels, and model.bias
% (You should need no for loops. This can be done in one line of code.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,samples]=size(data);
score= sum((repmat(model.svLabels',samples,1)).*(repmat(model.alphas',samples,1)).*(gramMatrix),2)+model.bias; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of score computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels = 2*(score(:) > 0) - 1; % threshold and map to {-1, 1}


