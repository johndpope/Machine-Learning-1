function gain = calculateInformationGain(data, labels)

% Computes the information gain on label probability for each feature in data
% data: d x n matrix of d features and n examples
% labels: n x 1 vector of class labels for n examples
% gain: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))

% REPLACE THE FOLLOWING WITH YOUR IMPLEMENTATION

totalEntropy = calculateEntropy(labels);

[d,n]=size(data);
gain = zeros(d,1);


%interate over all dimensions
for i=1:d
    %sort Xi and calulate probability of each element
    x_i= data(i,:);
    hist_x=tabulate(x_i);
    hist_x=[(cast(cell2mat(hist_x(:,1)),'double')-48),cell2mat(hist_x(:,2:3))];
    [count,~]=size(hist_x);
    scale=[ones(count,2),100.*ones(count,1)];
    hist_x=hist_x./scale;
   
  
    %classify set y into y1 y2 y3.. based on dimension |x_i|
    entropy= zeros(count,1);
    
    for k=1:count
       Y1 = find(x_i==hist_x(k,1));
       Y1=transpose(Y1);
       Y1(:,1)=labels(Y1(:,1),1);
       entropy(k,1)= calculateEntropy(Y1);
    end   
        
     prob=hist_x(:,3);
     informationGain= (transpose(prob)*entropy);
    informationGain= totalEntropy-informationGain;
    
     
     gain(i,1)=informationGain;
     
end
%calculate entropy(y1) and entropy (y2)...
    
    







