function model = mlpTrain(trainData, trainLabels, params, W0)

numHiddenUnits = params.numHiddenUnits;
numLayers = length(numHiddenUnits) + 1;

d = size(trainData, 1);
initVar = 1;

if exist('W0', 'var')
    % use given initialization
    model.W = W0;
else
    % initialize W randomly
    model.W = cell(numLayers, 1);
    model.W{1} = initVar * randn(numHiddenUnits(1), d + 1);
    for i = 2:numLayers-1
        model.W{i} = initVar * randn(numHiddenUnits(i), numHiddenUnits(i-1));
    end
    model.W{numLayers} = initVar * randn(1, numHiddenUnits(end));
end

model.squashFunction = params.squashFunction;
objective = zeros(params.maxIter, 1);
for i = 1:params.maxIter
    
    [objective(i), grad] = mlpObjective(model, trainData, ...
        trainLabels, params.lossFunction);
    
    rate = 0.1 / sqrt(i);
    
    for j = 1:numLayers
        change = - (grad{j} + params.lambda * model.W{j});
        if any(isinf(change(:))) || any(isnan(change(:)))
            % hack to handle numerical issues
            change(isinf(change) | isnan(change)) = 0;
            fprintf('Warning: entries overflowed\n');
        end
        model.W{j} = model.W{j} + rate * change;
    end
%             
%     if mod(i, 50) == 0
%         plot(objective(1:i));
%         drawnow;
%     end
end

