function [C,qtrain,alphaCenters] = KSCcodebook(etrain,alpha)

[N,d]=size(etrain);
k=d+1;
         
betabin=sign(etrain);
[C,m,uniquecw]=unique(betabin,'rows');

cwsizes = zeros(length(m),1);
for i=1:length(m)
    cwsizes(i) = sum(uniquecw==i);
end;

[~,j]=sort(cwsizes,'descend');

if length(m)<k
    k = length(m);
end;


C = C(j(1:k),:);
qtrain = zeros(N,1);

for i=1:k
    qtrain(uniquecw==j(i))=i; 
end;

alphaCenters = zeros(k,d);


%alphaCenters = grpstats(alpha,qtrain,{'median'});

for i=1:k
    alphaCenters(i,:) = median(alpha(qtrain==i,:));
end;


% if size(alphaCenters,1)==k+1
%    alphaCenters = alphaCenters(2:end,:);   
% end;

qtrain =  KSCmembership(etrain,C,alpha,alphaCenters);   


     
     
     
        
        
    
    
    
    
    