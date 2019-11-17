function MultiVK_MHKS_Main1(data,file_id,inPutInf)

M=inPutInf.M;
CV=size(data,1);
time = zeros(CV,1);
regTotal=zeros(CV,1);
inPutInf.kernel={'rbf','rbf','rbf'};

%优化
avPar = zeros(CV,1);
for CVCycle = 1 : CV

    %%%% kddcup %%%%    
    trn{CVCycle} = [data{CVCycle, 1};data{CVCycle, 2}];
    test{CVCycle} = data{CVCycle, 3};
    indTrnP = find(trn{CVCycle}(:,end) == 1);
    indTrnN = find(trn{CVCycle}(:,end) == 2);
    train_classone_data{CVCycle} = trn{CVCycle}(indTrnP,1:end-1);
    train_classtwo_data{CVCycle} = trn{CVCycle}(indTrnN,1:end-1);
    
    %求fuzzy membership
    tic
%     FM{CVCycle}=getWeightMat(train_classone_data{CVCycle}(:,1:end-1),train_classtwo_data{CVCycle}(:,1:end-1));
    FM{CVCycle} = eye(size(trn{CVCycle},1)); % 退化为单位矩阵
    ftime = toc;
    temp_Data=[train_classone_data{CVCycle};train_classtwo_data{CVCycle}];
    total_train_num=size(temp_Data,1);
    inPutInf.kPar=cell(3,1);
    avPar(CVCycle)=aveRBFPar(temp_Data,total_train_num);
end
i_num=1;  %记录总的个数
%参数
for i_c=1:length(inPutInf.cConf)
    inPutInf.C=inPutInf.cConf(i_c)*ones(inPutInf.M,1);  %参数C
    for i_gamma=1:length(inPutInf.gammaConf)
        inPutInf.gamma=inPutInf.gammaConf(i_gamma);         %参数gamma
        for i_kPar=1:length(inPutInf.kParConf)-2
            for j_kPar=i_kPar+1:length(inPutInf.kParConf)-1
                for k_kPar=j_kPar+1:length(inPutInf.kParConf)
                    inPutInf.kParA=inPutInf.kParConf(i_kPar);   %核参1
                    inPutInf.kParB=inPutInf.kParConf(j_kPar);   %核参2
                    inPutInf.kParC=inPutInf.kParConf(k_kPar);   %核参3
                    for CVCycle = 1 : CV
                        tic;
                        inPutInf.kPar{1}=inPutInf.kParA*avPar(CVCycle);
                        inPutInf.kPar{2}=inPutInf.kParB*avPar(CVCycle);
                        inPutInf.kPar{3}=inPutInf.kParC*avPar(CVCycle);
                        
                        subResult=MultiVK_MHKS_Fun(train_classone_data{CVCycle},train_classtwo_data{CVCycle},inPutInf,FM{CVCycle});
                        resultTrain.T=subResult.T;
                        resultTrain.w=subResult.w;
                        
                        time(CVCycle) = toc;%?ktimes?MCCV????
                        X=cat(1,train_classone_data{CVCycle},train_classtwo_data{CVCycle});
                        Z = test{CVCycle}(:,1:end-1);  %测试样本
                        totalSizeTest=size(Z,1);
                        resultMat=zeros(totalSizeTest,1);                        
                        Y_test=test{CVCycle}(:,end);                        
                        u=inPutInf.u;
                        indTrnP = find(trn{CVCycle}(:,end) == 1);
                        indTrnN = find(trn{CVCycle}(:,end) == 2);
                        Y=[ones(length(indTrnP),1);-1*ones(length(indTrnN),1)];
                        tempResult=0;
                        kName = inPutInf.kernel;
                        kPar=inPutInf.kPar;
                        for p=1:M
                            kernelTestMat_multiV{p}=GeKernel(X,Z,kName{p},kPar{p});
                            tempResult=tempResult+u(p)*((resultTrain.T{p}.*Y)'*...
                                kernelTestMat_multiV{p}+resultTrain.w{p});
                        end;
                        clear kernelTestMat_multiV;
                        
                        result=sign(tempResult);
                        tempVector1=find(result==1);
                        resultMat(tempVector1)=1;
                        tempVector2=find(result==-1);
                        resultMat(tempVector2)=2;
                        clear tempVector1;
                        clear tempVector2;
                        
                        totalRecog=length(find(Y_test==resultMat))/totalSizeTest;
                        reg=totalRecog*100;
                        regTotal(CVCycle)=reg;
                        
                        Res_tst_acc_P = length(find(Y_test(find(Y_test==1)) == resultMat(find(Y_test==1))))/length(find(Y_test==1));
                        Res_tst_acc_N = length(find(Y_test(find(Y_test==2)) == resultMat(find(Y_test==2))))/length(find(Y_test==2));
                        Res_tst_AUC = 0.5*(Res_tst_acc_P + Res_tst_acc_N);
                        aucTotal(CVCycle)=Res_tst_AUC;
                        Res_tst_GM = sqrt(Res_tst_acc_P*Res_tst_acc_N);
                        gmTotal(CVCycle)=Res_tst_GM;
                        
                        fprintf('The %d cycle--- AUC: %f;\n',CVCycle,Res_tst_AUC);
                        fprintf(file_id,'The %d cycle--- AUC: %f; \r\n',CVCycle,Res_tst_AUC);
                    end%for_CVCycle
                    avAuc=mean(aucTotal);
                    avstd = std(aucTotal);
                    regResult(i_num,:)=[inPutInf.cConf(i_c),inPutInf.gammaConf(i_gamma),inPutInf.kParConf(i_kPar)...
                        ,inPutInf.kParConf(j_kPar),inPutInf.kParConf(k_kPar),avAuc,avstd];
                    i_num=i_num+1;
                    fprintf('The average of AUC: %f; the gamma is: %f; the average running time is: %f;\n'...
                        ,avAuc,inPutInf.gamma,sum(time)/CV);
                    fprintf(file_id,'The average of AUC: %f; the gamma is: %f; the average running time is: %f;\r\n'...
                        ,avAuc,inPutInf.gamma,sum(time)/CV);
                end%for_k_kPar
            end%for_j_kPar
        end%for_i_kPar
    end%for_i_gamma
end%for_i_c

allReg=regResult(:,6);
[maxReg,i_row]=max(allReg);
fprintf('The best of AUC is: %f; std: %f; the time is: %f;  the C is: %f;  the gamma is: %f; the kernel pars are: %f,%f,%f \n',...
    maxReg,regResult(i_row,7), ftime+mean(time), regResult(i_row,1),regResult(i_row,2),regResult(i_row,3),regResult(i_row,4),regResult(i_row,5));
fprintf(file_id,'The best of AUC is: %f; std: %f; the time is: %f; the C is: %f;  the gamma is: %f; the kernel pars are: %f,%f,%f \r\n',...
    maxReg,regResult(i_row,7), ftime+mean(time), regResult(i_row,1),regResult(i_row,2),regResult(i_row,3),regResult(i_row,4),regResult(i_row,5));
end