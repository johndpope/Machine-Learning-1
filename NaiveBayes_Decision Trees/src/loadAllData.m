% script loads data from 20new-bydate

trainDataIJV = load('../20news-bydate/matlab/train.data');
testDataIJV = load('../20news-bydate/matlab/test.data');

maxWord = max([trainDataIJV(:,2); testDataIJV(:,2)]);

trainData = sparse(trainDataIJV(:,2), trainDataIJV(:,1), trainDataIJV(:,3),...
    maxWord, max(trainDataIJV(:,1)));
testData = sparse(testDataIJV(:,2), testDataIJV(:,1), testDataIJV(:,3),...
    maxWord, max(testDataIJV(:,1)));

% convert all data to Boolean values
trainData = trainData > 0;
testData = testData > 0;

trainLabels = load('../20news-bydate/matlab/train.label');
testLabels = load('../20news-bydate/matlab/test.label');


% this next block of code assumes the training and testing maps are the same
fid = fopen('../20news-bydate/matlab/train.map', 'r');
map = textscan(fid, '%s %d');
map = map{1}; % this assumes the categories are numbered 1 through k
% map = map{1}(map{2}); % use this if the map is arbitrarily ordered
fclose(fid);

