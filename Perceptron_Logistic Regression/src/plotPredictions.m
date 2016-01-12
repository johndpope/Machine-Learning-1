function [] = plotPredictions(X, Y, predictions)

% Plots 4-class 2-d data, indicating which points are in which class and
% which are classified correctly

numClasses = max(Y);

markers = {'x', 'o', '*',  'd'};

hold on;
for i = 1:numClasses
    plot(X(1, Y == i & Y == predictions), X(2, Y == i & Y == predictions),...
        [markers{i} 'g'], 'MarkerSize', 12, 'LineWidth', 1);
    plot(X(1, Y == i & Y ~= predictions), X(2, Y == i & Y ~= predictions),...
        [markers{i} 'r'], 'MarkerSize', 12, 'LineWidth', 1);
end
hold off;


