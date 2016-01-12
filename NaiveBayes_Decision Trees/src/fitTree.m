function node = fitTree(Data,depth,depthMax,data_y)

%%check end cases
   node = struct('value',0,'nodeleft',0,'noderight',0);
  %find y from data
  [size_y,~]=size(unique(data_y));
  if size_y == 1 || depth==depthMax
     [val,~]=mode(data_y);
     node.value=val;
     return
  end

  
  %%create node;
  
%% compute dimension to split
    w=calculateInformationGain(Data,data_y);
    [~,split_dimension]=max(w);
   node.value=split_dimension;
   
%%split on dimension
    set_yl=find(Data(split_dimension,:)==0);
    set_yr=find(Data(split_dimension,:)==1);
   
    
    dataL=Data(:,set_yl);
    dataR=Data(:,set_yr);
    
    yL= data_y(set_yl,:);
    yR= data_y(set_yr,:);
    
    
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
node.nodeleft=fitTree(dataL,depth+1,depthMax,yL);
node.noderight=fitTree(dataR,depth+1,depthMax,yR);

