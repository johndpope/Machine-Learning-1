function [means, sigma, probs] = gmm(data, numClusters)

% computes the cluster probabilities, means, and covariances of a Gaussian
% mixture model
% data: d x n matrix of n d-dimensional data points
% numClusters: scalar count of number of Gaussians to fit
% means: d x numClusters matrix of mean vectors
% sigma: length numClusters cell array of d x d covariance matrices
% probs: k x 1 multinomial probability vector for cluster membership

%data=data(1:2,:);
[d,n] = size(data);

%% initialize clusters

means = 0.01 * randn(d, numClusters);
sigma = cell(numClusters,1);

for i = 1:numClusters
    sigma{i} = eye(d);
end

probs = ones(numClusters,1) / numClusters;

%% start outer loop

maxIters = 1000;
tolerance = 1e-4;

prevProbs = 0;

% add this to covariance matrices to prevent them from getting too skinny
reg = 1e-4 * eye(d); 
for i = 1:maxIters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Insert your code to update cluster membership probabilities
    % for each data point
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Z = zeros( numClusters, n );
   
    for j = 1:numClusters
       Z(j,:) = gaussianLL(data, means(:,j), sigma{j}) + log(probs(j));   
    end
    
    Z = exp(Z - ones(numClusters, 1) * logsumexp(Z));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Insert your code to update cluster prior, means, and covariances
    % (probs, means, sigma)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    probs = mean(Z, 2);
    
    means = data * Z' ./ (ones(d,1) * sum(Z, 2)');
    
    for j = 1:numClusters
        whiten = data - means(:,j) * ones(1, n);
        sigma{j} = reg + ...
            ((whiten .* (ones(d, 1) * Z(j,:))) * whiten') / sum(Z(j,:));
    end

    
    % plot GMM
    
    clf;
    hold on;
    plot(data(1,:), data(2,:), 'x');
    plotGMM(means, sigma);
    hold off;
    title(sprintf('GMM with %d Clusters', numClusters), 'FontSize', 12);
    drawnow;
    
    % check for convergence
    
    change = norm(prevProbs - probs);
    prevProbs = probs;
    if change < tolerance
        break;
    end
end
