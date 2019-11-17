function ytest=membership(ytest,C)

if ~iscell(ytest)
    if size(C,1)==size(C,2) %oneVsALL
        ytest2=sign(ytest);
        [idx,D] = knnsearch(C,ytest2,'distance','hamming','k',size(C,2));
        ytest = idx(:,1);
    else %MOC
        ytemp = ytest;
        ytest = code_toolbox(ytemp,[1:size(C,2)],[],C);
    end
else
    Nviews = length(ytest);
    ytmp = ytest;
    ytest = cell(size(ytest));
    for v=1:Nviews
        ytest{v} = membership(ytmp{v},C);
    end
end

