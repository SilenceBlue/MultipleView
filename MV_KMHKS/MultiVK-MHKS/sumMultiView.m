function value=sumMultiView(K_multi,T,w,label,M,u)

value=0;
for p=1:M;
    value=value+u(p)*(K_multi{p}*T{p}+w{p}*label);
end;