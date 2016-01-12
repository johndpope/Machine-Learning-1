function entropy= calculateEntropy(labels)

[n , ~]=size(labels);

y=tabulate(labels);
y(y(:,2)==0,:)=[];
y=y(:,2);
y=(y./n).*log2(y./n);
entropy=sum(y);
entropy=-entropy;