function disp_nmi_model(model,Ytr,Yt)

qtr = model.qtrain;


if(iscell(qtr))
    Nviews = length(qtr);
    try
        nmitr=[];
        for v=1:Nviews
            nmitrv = nmi(Ytr{v},qtr{v});
            nmitr = [nmitr;nmitrv];
        end
        nmitr = mean(nmitr);
        disp(['nmi training: ' num2str(nmitr)]);
    catch e
        disp(['nmi training errror ' e.message]);
    end
    
    try
        qt = model.qtest;
            if exist('Yt','var') && ~isempty(Yt)
                try
                    nmit=[];
                    for v=1:Nviews
                        nmitv = nmi(Yt{v},qt{v});
                        nmit = [nmit;nmitv];
                    end
                    nmit = mean(nmit);
                    disp(['nmi test: ' num2str(nmit)]);
                catch e
                    disp(['nmi test errror ' e.message]);
                end
            end
    catch
        %no test set
    end
else
    try
        nmitr = nmi(Ytr,qtr);
        disp(['nmi training: ' num2str(nmitr)]);
    catch e
        disp(['nmi training errror ' e.message]);
    end
    
    try
        qt = model.qtest;
            if exist('Yt','var') && ~isempty(Yt)
                try
                    nmit = nmi(Yt,qt);
                    disp(['nmi test: ' num2str(nmit)]);
                catch e
                    disp(['nmi test errror ' e.message]);
                end
            end
    catch
        %no test set
    end
end