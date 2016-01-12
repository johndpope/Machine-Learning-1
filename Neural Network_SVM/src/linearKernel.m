function gramMatrix = linearKernel(dataI, dataJ)

% computes the linear kernel function for every pair of data across two data sets
% dataI - d x m data set to form the rows of the Gram matrix
% dataJ - d x n data set to form the columns of the Gram matrix
% gramMatrix - m x n matrix where the (i,j)th entry is k(x_i, x_j)

gramMatrix = dataI'*dataJ;
