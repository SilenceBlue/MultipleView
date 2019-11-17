function qtest=KSCmembership(etest,C,alphat,alphaCenters)

nt=size(etest,1);

etest2=sign(etest);


[idx,D] = knnsearch(C,etest2,'distance','hamming','k',size(C,2));

qtest = idx(:,1);

if size(D,2)>1

    nidx = find(D(:,2)==D(:,1));   
    
    [~,qtest(nidx)] = min(pdist2(alphat(nidx,:),alphaCenters),[],2);
    
    
%     for i=1:numel(nidx)     
%             idxmindist = idx(nidx(i),D(nidx(i),:)==D(nidx(i),1));
%             [~,minidx]=min(pdist2(alphat(nidx(i),:),alphaCenters(idxmindist,:)));
%             qtest(nidx(i)) = idxmindist(minidx);
%             mqtest{nidx(i)} = idxmindist;        
%     end;

end;






