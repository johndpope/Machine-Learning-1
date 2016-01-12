function ll = gmmLL(data, means, sigma, probs)

% computes the overall likelihood of data given a Gaussian mixture model
% data: d x n matrix of column examples
% means: d x k matrix of k means
% sigma: length k cell array of d x d covariance matrices
% probs: k x 1 vector of cluster membership prior probabilities
% 
% The formula should be \prod_i \sum_k p(k) p(x_i | k)
% The log of that is \sum_i \log(sum_k p(k) p(x_i | k))
% p(x_i | k) should come from gaussianLL.m

[~,numClusters]=size(means);
[~,n]=size(data);
Z = zeros( numClusters, n );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code below to compute the log likelihood of data
% under a Gaussian mixture model.
% Your code should use gaussianLL.m, which computes the Gaussian 
% log-likelihood of a set of data vectors 
% You will also want to use logsumexp.m to avoid numerical issues 
% with very small probability values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:numClusters
    
    Z(i,:) = gaussianLL( data, means(:,i), sigma{i});
    
end

ll = sum(logsumexp(Z + log(probs) * ones(1, n)));

end


