
%calculates accuracy of single run

function score= accuracy(prediction,labels)
count=0;
    [size_y,~]=size(prediction);
    for k=1:size_y
        if labels(k,1)==prediction(k,1)
            count=count+1;
        end
    end
    
    score=count/size_y;