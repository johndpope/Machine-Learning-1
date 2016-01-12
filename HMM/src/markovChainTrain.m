function model = markovChainTrain(sequence, alpha)

% trains a Markov chain model based on an 
% input sequence
% sequence: index array of ordered variable states
% alpha: prior term for cpts and prior
% model should contain two main structures:
%   model.prior: |dict| x 1 vector of prior probabilities p(state)
%   model.transitions: |dict| x |dict| matrix of conditional probabilities
%          p(next state | current state) = model.transitions(currentState, nextState)
%          i.e., where each i'th row is the conditional probability
%          distribution of the next state given that the current state
%          is i

numStates = max(sequence);
n = length(sequence);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 of 2
% Insert your code here for fitting model.prior
% Hint: for both types of model fitting, using 
% sparse indexing can be much faster than 
% iterating over the full sequence.
% Read the documentation with 'help sparse'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

counts = sparse(sequence, ones(n,1), ones(n,1));
model.prior = (full(counts) + alpha) / (sum(counts) + numStates * alpha);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 of 2
% Insert your code here for fitting model.transitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

counts = sparse( sequence( 1 : end-1 ), sequence( 2 : end ), ones( n - 1, 1 ) );

model.transitions = (full(counts) + alpha) ./ (sum(counts, 2) * ones(1, numStates) + alpha * numStates);

end