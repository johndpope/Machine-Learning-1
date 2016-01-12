function label = naiveBayesPredict(data, model)

% FILL IN YOUR CODE AND COMMENTS HERE
%for each sample in data

    [d,n]=size(data);
    [~,l]=size(model);
    prediction=zeros(n,l);
    label=zeros(n,1);
    for sample_no=1:n
        for j=1:l
            sumx=sum(data(:,sample_no).*model(:,j));
            prediction(sample_no,j)= sumx;
        end
        [~,label(sample_no,1)]= max(prediction(sample_no,:));
    end


 % replace this with your actual lbel

