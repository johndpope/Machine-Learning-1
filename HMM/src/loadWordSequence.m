function [wordIndex, words] = loadWordSequence(filename)

% loads a text document as a sequence of words
% treats a newline as a word
% filename: path to input file
% wordIndex: index array, where wordIndex(i) is the
%            index in 'words' of the i'th word in
%            the document
% words: dictionary of words


fid = fopen(filename);
lines = textscan(fid,'%s','delimiter','\n');
allText = strjoin(lines{1}', '\n');

[words, ~, wordIndex] = unique(strsplit(allText, ' '));
words = words';

