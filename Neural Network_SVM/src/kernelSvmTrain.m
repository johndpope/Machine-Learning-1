function model = kernelSvmTrain(trainData, trainLabels, params)

% trains a kernel SVM using one of the three preset kernel types
% trainData - d x n matrix of n data vectors
% trainLabels - n x 1 vector of {-1, 1} binary labels
% params - struct containing 
%   params.kernel - either 'linear', 'polynomial', or 'rbf'
%   params.sigma - variance parameter for rbf
%   params.order - polynomial order for polynomial kernel
%   params.C - slack parameter for SVM objective

% Compute the gram matrix for the selected kernel
if isfield(params, 'kernel') && strcmp(params.kernel, 'rbf')
    gramMatrix = rbfKernel(trainData, trainData, params.sigma);
elseif isfield(params, 'kernel') && strcmp(params.kernel, 'polynomial')
    gramMatrix = polynomialKernel(trainData, trainData, params.order);
else
    % use a linear kernel by default
    gramMatrix = linearKernel(trainData, trainData);
end

gramMatrix = (gramMatrix + gramMatrix') / 2; % correct any minor numerical errors

n = size(gramMatrix,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code below to compute the inputs to the quadratic program
% solver, which minimizes 
% min 0.5*x'*H*x + f'*x  
%  x
% subject to:  A*x <= b
%              Aeq*x = beq
%              lowerBounds <= x <= upperBounds
% Compute the variables H, f, A, b, Aeq, beq, lowerBounds, and upperBounds
% If you don't need any of the constraint types, you can assign them to be
% empty. E.g., A = [] and b = [] (this is a hint.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create a dxn matrix of trainLabels
[~,samples]=size(trainData);
y=trainLabels*trainLabels';
H= y.*gramMatrix;

%x is alpha matrix of order samples X 1 

f=ones(samples,1)*(-1);
%A*x<=b condition not needed
A=[];
b=[];

%summation of all alpha should be zero
Aeq=trainLabels';
beq=0;
%bounds from 0 to C
lowerBounds=zeros(samples,1);
upperBounds=ones(samples,1)*params.C;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of quadprog setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');

alphas = quadprog(H, f, A, b, Aeq, beq, lowerBounds, upperBounds, [], opts);
% only store support vectors and alphas that have nonnegligible support
tolerance = 1e-6;

svIndices = alphas > tolerance;
%svIndices=ones(samples,1);

model.sv = trainData(:, svIndices);

model.alphas = alphas(svIndices);
model.params = params; % store the kernel type and parameters
model.svLabels = trainLabels(svIndices);

marginAlphas = (alphas > tolerance) & (alphas < params.C - tolerance);

model.bias = mean(trainLabels(marginAlphas)' - ...
    (alphas .* trainLabels)'*gramMatrix(:, marginAlphas));

% check if there are no margin support vectors
if isnan(model.bias)
    model.bias = 0;
end
