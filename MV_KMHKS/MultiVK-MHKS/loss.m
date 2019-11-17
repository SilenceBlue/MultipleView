function value=loss(K_multi,T,w,label,B,C,gamma,totalSize,u,M,FM)

tempValue1=0;
tempValue2=0;
SMV=sumMultiView(K_multi,T,w,label,M,u);

for p=1:M;
    tempValue1=tempValue1+(K_multi{p}*T{p}+w{p}*label-ones(totalSize,1)-B(:,p))'*FM*(K_multi{p}*T{p}+w{p}*label-ones(totalSize,1)-B(:,p))...
        +C(p)*T{p}'*K_multi{p}*T{p};    
    tempValue2=tempValue2+(K_multi{p}*T{p}+w{p}*label-SMV)'*FM*(K_multi{p}*T{p}+w{p}*label-SMV);
end;

value=tempValue1+gamma*tempValue2;