clear;
clc;
% load .\DataSets\Imbalanced_data.mat;

inPutInf.M=3; % 目前暂时固定为3
%参数设置
% inPutInf.cConf=[1e-3,1e-2,1e-1,1,10];  %参数C集
inPutInf.cConf=0.001;  %参数C集
inPutInf.R=0.99*ones(inPutInf.M,1);
inPutInf.B=1e-6*ones(inPutInf.M,1);
inPutInf.sizeIter=100;
inPutInf.termination=1e-3;
inPutInf.u=ones(inPutInf.M,1)/inPutInf.M;
% inPutInf.gammaConf=[1e-3,1e-2,1e-1,1,10];  %参数gamma集
inPutInf.gammaConf=10;  %参数gamma集
% inPutInf.gammaConf=2^(-4);  %参数gamma集
inPutInf.u=ones(inPutInf.M,1)/inPutInf.M;
% inPutInf.kParConf=[2^(-3),2^(-2),2^(-1),1,2,4,8];   %核参数集
inPutInf.kParConf=[0.125,0.25,8];   %核参数集

dataSetName= {
    'ecoli1';
    'glass0';
    'yeast1'
};
dirctory = './datasets/';
for dataId = 1:size(dataSetName)
    dataName = dataSetName{dataId};
    file_name=['./report/','CFMKL_',dataName,'.txt'];
    dataDir = strcat(dirctory, dataName);
    load(dataDir);
    fprintf('Current data: %s \n', dataName);
    n = strcat('=', dataName);
    fromName = strcat(n,';');
    toName = 'data';
    eval([toName fromName]);
    clear(dataName)
    
    file_id = fopen(file_name, 'w');
    fprintf('Setting: Dataset-%s \r\n',dataName);
    fprintf(file_id, 'Setting: Dataset-%s \r\n',dataName);
    MultiVK_MHKS_Main1(data,file_id,inPutInf);                 
    fclose(file_id);
    clear file_id;
end