function yO=code(yI,C)

nc = size(C,1);
if ~iscell(yI)
    if size(C,1)==size(C,2) %oneVsALL
        yO = -ones(size(yI,1),nc);
        for n=1:nc
           k=find(yI==n);
           yO(k,n)=1;
        end
    else %MOC
        yO = code_toolbox(yI,'code_MOC');
    end
else
    yO = cell(size(yI));
    Nviews = length(yI);
    for v=1:Nviews
        yO{v} = code(yI{v},C);
    end
end

