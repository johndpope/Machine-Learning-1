function [] = plotData(data, labels)

% plots binary class data
% data - d x n data matrix of n examples
% labels - n x 1 binary class labels in {-1, 1}

plot(data(1, labels > 0), data(2, labels > 0), 'dk', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'm');
if ~ishold
    holdWasOff = true;
    hold on;
else
    holdWasOff = false;
end
plot(data(1, labels < 0), data(2, labels < 0), 'ok', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'c');
if holdWasOff
    hold off;
end
