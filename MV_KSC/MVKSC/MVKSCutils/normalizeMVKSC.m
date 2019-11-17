function Xnew = normalizeMVKSC(X)

if iscell(X)

    Xnew = cell(size(X));
    Nviews = size(X,2);
    
    for v=1:Nviews
        %Normalize zero mean, unit variance
        Xv=X{v}./ repmat(std(X{v}),size(X{v},1),1);
        Xnew{v} = Xv;
        Xv=Xnew{v}- repmat(mean(Xnew{v}),size(Xnew{v},1),1);
        Xnew{v} = Xv;
    end
    
else
    
    %Normalize zero mean, unit variance
    Xnew=X./ repmat(std(X),size(X,1),1);
    Xnew=Xnew- repmat(mean(Xnew),size(Xnew,1),1);
    
end