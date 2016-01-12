rng(0);

n = 2000;

trueDim = 2;
dim = 64;
numClusters = 5;

%% set up Gaussian mixture parameters

means = randn(trueDim, numClusters);

sigma = cell(numClusters, 1);

for k = 1:numClusters
    x = randn(trueDim, trueDim);
    sigma{k} = x * x';
end

classProb = sqrt(rand(numClusters, 1));
classProb = classProb / sum(classProb);

%% sample cluster memberships

inCluster = sampleMultinomial(classProb, n);

%%

trueData = zeros(trueDim, n);

for i = 1:numClusters
    I = inCluster == i;
    
    trueData(:,I) = sampleGaussian(means(:,i), sigma{i}, sum(I));
end

%% plot data and Gaussians

clf;
hold on;
plot(trueData(1,:), trueData(2,:), '.');
plotGMM(means, sigma);
hold off;


%% create higher dimensional representation and add noise

map = randn(dim, trueDim);

noise = 3;

data = map * trueData + noise * randn(dim, n);

%% save to mat files

save ../trueData trueData means sigma classProb;
save ../synthData data;

