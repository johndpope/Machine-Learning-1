function model = myHmmTrain(trainData, params)

% trainData: d x T matrix of observation columns
%   params.numStates: number of states for hidden variables
%   model.prior: numStates x 1 vector of prior probabilities p(state)
%   model.transitions: numStates x numStates matrix of conditional probabilities
%          p(next state | current state) = model.transitions(nextState, currentState)
%          i.e., where each i'th column is the conditional probability
%          distribution of the next state given that the current state
%          is i
%   model.means: d x numStates matrix of mean columns for Gaussian
%               probabilities of p(data | state)
%   model.sigma: length numStates cell array of d x d covariance matrices
%                for p(data | state)

[d,T] = size(trainData);

numStates = params.numStates;
if isfield(params, 'maxIter')
    maxIter = params.maxIter;
else
    maxIter = 100;
end
if isfield(params, 'tolerance')
    tolerance = params.tolerance;
else
    tolerance = 1e-6;
end
if isfield(params, 'draw')
    draw = params.draw;
else
    draw = false;
end
if isfield(params, 'alpha')
    alpha = params.alpha;
else
    alpha = 0;
end

% numerical constants
reg = 1e-4 * eye(d);

% initialize model

model.prior = rand(numStates,1);
model.prior = model.prior / sum(model.prior);

model.transitions = rand(numStates, numStates);
model.transitions = model.transitions ./ ...
    (ones(numStates, 1) * sum(model.transitions));

% fit Gaussian to full data
mu = mean(trainData, 2);
diff = trainData - mu * ones(1,T);
sigma = diff * diff';

model.means = sampleGaussian(mu, sigma, numStates);
model.sigma = cell(numStates, 1);
for j = 1:numStates
    % use full data covariance matrix
    diff = trainData - model.means(:,j) * ones(1,T);
    model.sigma{j} = diff * diff' + reg;
end

% main EM loop

prevLL = -inf;

for i = 1:maxIter
    % E-step
    [pState, ll, pTransitions] = myHmmInferStates(trainData, model);
    
    % M-step
    model.prior = alpha + sum(pState, 2);
    model.prior = model.prior / sum(model.prior);
    model.transitions = alpha * ones(numStates, numStates);
    for j = 1:T-1
        model.transitions = model.transitions + pTransitions{j};
    end
    model.transitions = model.transitions ./ (sum(model.transitions, 2) * ones(1, numStates));
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1 of 1
    % Insert your code here to compute the updated model.means and model.sigma
    % Hint: the construction X * spdiag(p) * X' may be useful for
    % vectorizing a certain sum of outer products that is part of the
    % update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:numStates
        model.means(:,j) = trainData * pState(j,:)' / sum(pState(j,:));
        diff = trainData - model.means(:,j) * ones(1,T);
        model.sigma{j} = bsxfun(@times, diff, pState(j,:)) * diff' / sum(pState(j,:)) + reg;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % End of required section
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % plot and output some diagnostics
    if mod(i, 5) == 0
        fprintf('Iteration %d, log likelihood %d. Change %d\n', i, ll, abs(ll - prevLL));
        if d == 2 && draw
            clf;
            subplot(311);
            plot(trainData(1,:), trainData(2,:), '.');
            hold on;
            plotGMM(model.means, model.sigma);
            hold off;
            title('Observation Gaussians');
            subplot(312);
            imagesc(model.transitions);
            colorbar;
            title('Transition Matrix');
            subplot(313);
            bar(model.prior);
            title('Prior');
            drawnow;
        end
    end
    
    % check for convergence
    if abs(prevLL - ll) < tolerance
        fprintf('Log likelihood change was below tolerance. Learner seems to have converged.\n');
        break;
    end
    prevLL = ll;
    
end