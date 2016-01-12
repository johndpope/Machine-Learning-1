function acc= Accuracy(y,y_predict)

[samples,~]=size(y);
acc=0;
for i = 1:samples
    if y(i,1)==y_predict(i,1)
        acc=acc+1;
    end
end
acc=acc/samples;
end