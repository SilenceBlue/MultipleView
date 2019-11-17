function acc=get_acc(yhatt,Yt)

if iscell(yhatt)
    Nviews = length(yhatt);
    
    acct = zeros(1,Nviews);
    for v=1:Nviews
        acct(v) = 100*sum(yhatt{v}==Yt)/length(Yt);
    end
    acc=mean(acct);
else
    acc = 100*sum(yhatt==Yt)/length(Yt);
end