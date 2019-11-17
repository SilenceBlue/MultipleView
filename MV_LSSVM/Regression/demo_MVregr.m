%Demo studying the performance of the MV-LSSVM method on weather
%forecasting

%Author: Lynn Houthuys

%Citation: 

%Houthuys L., Karevan Z., Suykens J.A.K., "Multi-view LS-SVM Regression 
%for Black-Box Temperature Prediction in Weather Forecasting", in Proc. 
%of the International Joint Conference on Neural Networks (IJCNN), Anchorage, 
%USA, May 2017, pp. 1102-1108.

% Dataset : www.wunderground.com

% April min. temp. 2 days ahead
load('DataAprStepAhead2.mat');
load('Params2Apr.mat');
Nviews=10;
kernel='RBF_kernel';


%prep data
for j=1:Nviews
    Dtotal=[];
    Ttotal=[];
    for i=1:lag
        D1=[Data{i,j}(1:end-28,:)];
        T1=[Data{i,j}(end-27:end,:)];
       
        Dtotal=[Dtotal,D1];
        Ttotal=[Ttotal,T1];
    end
    Xtr{j}=Dtotal;
    Xtests{j}=Ttotal;
end

%mean
model=MVRegr(Xtr,Ytrain,kernel,Sigmas,Gammas,rho_mean,'eye','mean',Xtests);
mae_tmp = zeros(1,Nviews);
for v=1:Nviews
    mae_tmp(v) = mae(Ytest - round(model.yhat_test{v}));
end
maet = mean(mae_tmp);
disp(['test MAE April 2 days ahead MVLS-SVM mean: ' num2str(maet)]);

%median
model=MVRegr(Xtr,Ytrain,kernel,Sigmas,Gammas,rho_median,'eye','median',Xtests);
mae_tmp = zeros(1,Nviews);
for v=1:Nviews
    mae_tmp(v) = mae(Ytest - round(model.yhat_test{v}));
end
maet = mean(mae_tmp);
disp(['test MAE April 2 days ahead MVLS-SVM median: ' num2str(maet)]);