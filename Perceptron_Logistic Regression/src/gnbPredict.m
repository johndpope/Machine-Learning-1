function label = gnbPredict(data, model)

% predicts most likely label given computed means and class prior 
% data: single d x n matrix of n data points to be classified
% model: model object from naiveBayesTrain


[d,samples]=size(data);
[~,classes]=size(model.mean);
label=zeros(samples,1);
out=zeros(classes,1);

for i=1:samples
    for j=1:classes
        prob=model.prior(j,1);
        prob=log(prob);
        
        secondterm =(1/(2*model.sigma^2)) ...
                    *transpose(data(:,i)-model.mean(:,j)) ...
                    *(data(:,i)-model.mean(:,j));
        out(j,1)=prob-secondterm;
    end
    [~,predict]=max(out);
    label(i,1)=predict;
end
    
    
    
    
        
