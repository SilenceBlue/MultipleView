%Demo studying the performance of the MV-LSSVM method on a real-world dataset

%Author: Lynn Houthuys

%Citation: 

%L. Houthuys, R. Langone, and J. A. K. Suykens, Multi-View Least Squares Support Vector Machines
%Classification , Neurocomputing, vol. 282, Mar. 2018, pp. 78-88.

% Dataset : T. Kolenda, L.K. Hansen, J. Larsen, O. Winther, Independent component analysis for
% understanding multimedia content, in: Proceedings of IEEE Workshop on Neural Networks for 
% Signal Processing, 12, 2002, pp. 757–766.

clear all;
rng default;
addpath('MVClassUtils');

%% Download data (only split 1)
load('Kolenda_split1');
nc=3;
N=size(X{1},1);
Nviews=3;

%% Settings - parameters obtained by tuning
Codebook = code_OneVsAll(nc);
kernel={'RBF_kernel','RBF_kernel','lin_kernel'};
rho=0.1682;
gam={0.2337,6.3415,0.1660};
sig={1.5446,25.1533,0};

%% Run algorithm
model = MVClass(X,code(Y,Codebook),kernel,sig,gam,rho,'mean',Xt);

%% Evaluate performance
yhatt = membership(model.yhat_test,Codebook);
acc=get_acc(yhatt,Yt)