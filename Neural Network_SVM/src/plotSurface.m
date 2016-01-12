function [] = plotSurface(predictor, model)

% plots the score surface of a model in the range [-2, 2]
% predictor - prediction function that takes a data matrix and model
% model - model struct compatible with the predictor function

x = linspace(-3, 3, 100);

data = zeros(2, length(x)^2);
count = 1;
for i = 1:length(x)
    for j = 1:length(x)
        data(:, count) = [x(i); x(j)];
        count = count + 1;
    end
end
[~, p] = predictor(data, model);

p = reshape(p, length(x), length(x));

% attempt to guess score range
if min(p(:)) >= 0
    contourf(x, x, p, 'LevelList', 0.5);
else
    contourf(x, x, p, 'LevelList', 0);
end


