function [pState, logLikelihood, pTransitions] = myHmmInferStates(data, model)

% performs the forward-backward algorithm to infer the latent state
% probabilities of a hidden Markov model
% data: d x T matrix of observations at each time step
% model: struct containing
%   model.means: d x numStates matrix of means of each state's observation
%                   model Gaussian
%   model.sigma: length-numStates cell array of dxd covariance matrices
%   model.prior: numStates x 1 vector of multinomial prior state prob
%   model.transitions: numStates x numStates matrix of conditional
%                   state-transition probabilities
% pState: numStates x T matrix of marginal state probabilities p(x_t | Y)
% logLikelihood: scalar marginal log likelihood of data
% pTransitions: marginal probabilities of all transitions p(x_t | x_{t-1})
%               represented as a cell array of matrices


[d, timeSteps] = size(data);
numStates = length(model.prior);

% precompute all Gaussian likelihoods for fast lookup

pObs = zeros(numStates, timeSteps);
for i = 1:numStates
    pObs(i,:) = exp(gaussianLL(data, model.means(:,i), model.sigma{i}));
end

% start of forward pass

pStateGivenPast = zeros(numStates, timeSteps);
pStateGivenPast(:,1) = model.prior .* pObs(:,1);

for t = 2 : timeSteps
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1 of 3
    % Insert your code to compute the forward pass to
    % compute pStateGivenPast(i,t) = p(x_t == i |y_1, ..., y_t, model)
    % Make sure to normalize to aid numerical stability and to get the
    % conditional probability.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pStateGivenPast(:,t) = pObs(:, t) .* (model.transitions' * pStateGivenPast(:, t-1));
    pStateGivenPast(:,t) = pStateGivenPast(:,t) / sum(pStateGivenPast(:,t), 1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % End of required section
    %%%%%%%%%%%%%%%%%%%%%%%%%%
end

% backward pass to compute p(y_{t+1}, ..., y_T | x_t, model)

pFutureObs = zeros(numStates, timeSteps);

pFutureObs(:, timeSteps) = 1;

for t = timeSteps:-1:2
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2 of 3
    % Insert your code to compute the backward pass to
    % compute pFutureObs(i,t) = p(y_{t+1}, ..., y_T | x_t == i, model)
    % Make sure to normalize to aid numerical stability
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    pFutureObs(:,t-1) = model.transitions * ( pObs(:, t) .* pFutureObs(:, t) );
    pFutureObs(:,t-1) = pFutureObs(:,t-1) / sum(pFutureObs(:,t-1), 1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % End of required section
    %%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 of 3
% insert your code to compute marginal probability of states given
% observations, pState(i,t) = p(x_t == i | Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pState = pStateGivenPast .* pFutureObs;
pState = pState ./ (ones(numStates, 1) * sum(pState, 1));

%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of required section
%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute log likelihood of data
logLikelihood = sum(log(sum(pState .* pObs, 1)));

% compute expected transitions
pTransitions = cell(timeSteps-1, 1);

for t = 1:timeSteps-1
    % compute the dxd expected transitions from i to i+1
    pTransitions{t} = model.transitions .* ...
        (pStateGivenPast(:,t) * (pObs(:,t+1) .* pFutureObs(:, t+1))');
    
    pTransitions{t} = pTransitions{t} / sum(pTransitions{t}(:));
end
