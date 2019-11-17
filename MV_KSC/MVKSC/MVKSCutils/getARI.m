function a = getARI(q,Y)

if iscell(q)
   Nviews = length(q);
   a=0;
   if iscell(Y)
       for v=1:Nviews
           a = a + adjrandindex(q{v},Y{v});
       end
       a=a/Nviews;
   else
       for v=1:Nviews
           a = a + adjrandindex(q{v},Y);
       end
       a=a/Nviews;
   end
else
    a = adjrandindex(q,Y);
end