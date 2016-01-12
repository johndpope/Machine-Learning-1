function [] = plotGMM(means, sigma)

% plots 2d Gaussian mixtures
% means: d x k matrix of k means 
% sigma: length k cell array of d x d covariance matrices

if ~ishold
    holdWasOff = true;
    hold on;
else
    holdWasOff = false;
end

[d, k] = size(means);

resolution = 50;

angles = linspace(0, 2*pi, resolution);

x = [cos(angles); sin(angles)];

for i = 1:k
    A = chol(sigma{i});
    ellipse = A * x + means(:,i) * ones(1, resolution);
    plot(ellipse(1,:), ellipse(2,:), 'r');
end

if holdWasOff
    hold off;
end
