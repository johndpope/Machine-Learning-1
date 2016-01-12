function gramMatrix = polynomialKernel(dataI, dataJ, order)

% Computes the polynomial kernel function for every pair of data across two data sets
% Computes the kernel without explicitly expanding the features
% dataI - d x m data set to form the rows of the Gram matrix
% dataJ - d x n data set to form the columns of the Gram matrix
% order - polynomial order to expand the features into
% gramMatrix - m x n matrix where the (i,j)th entry is k(x_i, x_j)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code below to compute the polynomial kernel between dataI and dataJ
% The most efficient way of computing this should only need one line of code 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gramMatrix=(dataI'*dataJ+1).^order;
