function label = decisionTreePredict(data, model)

% FILL IN YOUR CODE AND COMMENTS HERE
[~,n]=size(data);
label=zeros(n,1);
for i = 1:n
    currentnode=model;
    while ~isequal(currentnode.nodeleft,0)&&~isequal(currentnode.noderight,0)
        
        if data(currentnode.value,i)==1
            nextnode=currentnode.noderight;
        else
            nextnode=currentnode.nodeleft;
        end
        currentnode=nextnode;
    end
    label(i,1)=currentnode.value;
end
            
            

 % replace this with your actual label
