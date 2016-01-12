clear;

%% load historical NOAA data on Blacksburg

load ../blacksburgWeather.mat;
[d,T] = size(data);

%% plot dimensions

figure(1);
clf;
plot(data(:, 1:365)')

legend(variables);
ylabel('Value');
xlabel('Days since 01/01/82');

title(sprintf('First Year of Data (%2.2f Years Total)', T/365));

%% split data for validation

numYears = 8;

testInds = (1:T) > T - numYears * 365;

trainData = data(:, ~testInds);
testData = data(:, testInds);

%% train HMM

numStates = [2, 4, 6, 8];

learnedModels = cell(length(numStates),1);

params.alpha = 1e-3;
params.draw = false;

% we're using fairly loose parameters here
% because we're impatient
params.tolerance = 1e-4;
params.maxIter = 200;

figure(2);
clf;

% initialize likelihoods as nan to make plotting-as-we-go easier
logLikelihood = nan(length(numStates), 1);

for i = 1:length(numStates)
    params.numStates = numStates(i);
    learnedModels{i} = myHmmTrain(trainData, params);

    [~, logLikelihood(i)] = myHmmInferStates(testData, learnedModels{i});

    plot(numStates, logLikelihood);
    xlabel('Number of Hidden States');
    ylabel('Held-Out Log Likelihood');
    drawnow;
end

%% visualize HMM parameters

figure(3);
clf;
% hard-coded for 4-state HMM, which is easier to interpret
i = 2;

pStates = myHmmInferStates(testData, learnedModels{i});

subplot(211);
imagesc(pStates);
xlabel('Day');
ylabel('State Probability');
subplot(212);

subplot(223);

bar(learnedModels{i}.means(1:2,:)');
legend('Precipitation (mm)', 'Snow (mm)');
xlabel('State');
ylabel('mm');

subplot(224);

bar(learnedModels{i}.means(3:4,:)');
legend('High', 'Low');
ylabel('Celsius');
xlabel('State');
title('Mean Temperature Features');