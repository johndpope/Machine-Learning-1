function sequence = sampleMarkovChain(model, n)

% samples a sequence from a Markov chain model
% model: struct containing
%   model.prior: p(x_1)
%   model.transitions: numStates x numStates matrix of transition
%                       probabilities p(x_t | x_{t-1})
% sequence: n x 1 vector of states sampled from Markov chain model

sequence = zeros(n, 1);

sequence(1) = sampleMultinomial(model.prior, 1);

for i = 2:n
    sequence(i) = sampleMultinomial(model.transitions(sequence(i-1), :), 1);
end
