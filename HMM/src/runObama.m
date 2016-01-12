%% load the transcript of Obama's 2015 SOTU address
[sequence, words] = loadWordSequence('../obama15.txt');
% [sequence, words] = loadWordSequence('../toy.txt');


%% train a Markov chain on the speech text
alpha = 1e-6;
model = markovChainTrain(sequence, alpha);


%% generate data from the Markov chain
roboObama = sampleMarkovChain(model, 500);

fout = fopen('roboObama.txt', 'w');
fprintf(fout, strjoin(words(roboObama)'));
fclose(fout);
