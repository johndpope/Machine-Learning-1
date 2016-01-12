function gramMatrix = rbfKernel(dataI, dataJ, sigma)

% Computes the RBF kernel function for every pair of data across two data sets
% K(a,b) = exp((a - b)'*(a - b) / sigma)
% dataI - d x m data set to form the rows of the Gram matrix
% dataJ - d x n data set to form the columns of the Gram matrix
% sigma - variance parameter
% gramMatrix - m x n matrix where the (i,j)th entry is k(x_i, x_j)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code below to compute the Gaussian RBF kernel between dataI
% and dataJ. You should use the trick we used for the Gaussian naive Bayes
% model from HW2, which allowed us to compute the distance between all
% pairs of points in a vectorized fashion.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[d,m]=size(dataI);
[d,n]=size(dataJ);

dataIsq = diag(dataI'*dataI);      
if isempty(dataJ)
    dataJsq =ones(m,0)*transpose(diag(dataJ'*dataJ));
else
    dataJsq =ones(m,1)*transpose(diag(dataJ'*dataJ));
end
Val=dataIsq*ones(1,n)+dataJsq-2*dataI'*dataJ;


gramMatrix=exp(-Val/sigma );



