function [collinearity,balance]=BLF(projval,qval,dt)


MIN_CLUSTSIZE = 2;
MIN_CLUSTLAMBDA = 1e-9;

k = size(projval,2) + 1;
N = size(projval,1);

collinearity = zeros(k,1);
balance = 0;

uniq_qval = unique(qval);


csizes=zeros(length(uniq_qval),1);

for l = 1:length(uniq_qval)
    csizes(l) = sum(qval==uniq_qval(l));
end;

if length(uniq_qval) == k && uniq_qval(1) ~= 0 && min(csizes) >= MIN_CLUSTSIZE
    
    if k==2
        projval = [projval dt];
        projval = bsxfun(@minus,projval,mean(projval));
        projval = bsxfun(@rdivide,projval,std(projval));
        
    end;
    zetasmax = zeros(k,1);
    
    for l = 1:k
        cidx = qval == l;
        X = projval(cidx,:);
        
        zetas = sort(eig(cov(X)),'descend');
        
        if zetas(1)>MIN_CLUSTLAMBDA;
            
            if k==2
                collinearity(l) = 2*zetas(1)/sum(zetas)-1;
            else
                collinearity(l) = (k-1)/(k-2)*(zetas(1)/sum(zetas)-1/(k-1));
                
            end;
            
            
            
            zetasmax(l) = zetas(1);
        end;
        
    end;
    collinearity = collinearity.*zetasmax/max(zetasmax);
    csizeslog = log(csizes);
    
    balance = min(csizeslog)/max(csizeslog);
end

    



