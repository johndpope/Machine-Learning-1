function [newData, variances, eigenvectors] = myPca(data)

% performs PCA, moving the input data to a new basis and outputs the full
% rank representation with the captured variances for each dimension
% data: d x n data matrix of n d-dimensional examples
% newData: d x n new data matrix re-aligned to new basis
% variances: d x 1 vector of variance captured by each dimension
% eigenvectors: d x d matrix of eigenvectors (in columns)
% 
% The dimensions of newData and variances should be sorted so the first
% dimension captures the most variance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enter your code below for computing newData and variances. 
% You may use built in eig or svd, but you are not allowed to use the 
% built in pca in your implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~, n ] = size(data);

whiten = data - mean(data, 2) * ones(1, n);

[ ~, Sigma, eigenvectors ] = svd( whiten' );

variances = diag(Sigma).^2;

newData = eigenvectors' * data;

end





