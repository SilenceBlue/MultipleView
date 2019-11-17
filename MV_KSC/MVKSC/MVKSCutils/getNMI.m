function n = getNMI(q,Y)

if iscell(q)
   Nviews = length(q);
   n=0;
   if iscell(Y)
       for v=1:Nviews
           n = n + nmi(Y{v},q{v});
       end
       n=n/Nviews;
   else
       for v=1:Nviews
           n = n + nmi(Y,q{v});
       end
       n=n/Nviews;
   end
else
    n = nmi(Y,q);
end