function mat_kernel=GeKernel(mat_train,mat_test,kernelType,kPara)
% input parameters:
% mat_train--£­the train data, a row represents a sample
% mat_test---- the test data, a row represents a sample
% kernelType-- the kernel type
% kPara------- the kernel parameter(s)
% output parameter:
% mat_kernel-- the kernel matrix

% Written by WangZhe on 2004-09-27. 

switch lower(kernelType)
    case 'linear'
        mat_kernel=mat_train*mat_test';
    case 'poly'
        mat_kernel=(mat_train*mat_test'+1).^kPara;
    case 'rbf'
        TrainSampleNum=size(mat_train,1);
        TestSampleNum=size(mat_test,1);
        mat_temp=sum(mat_train.^2,2)*ones(1,TestSampleNum)...
                +ones(TrainSampleNum,1)*sum(mat_test.^2,2)'...
                -2*mat_train*mat_test';
        mat_kernel=exp(-mat_temp/(2*kPara^2));
    case 'sigmoid'
        mat_kernel=tan(kPara(1)*mat_train*mat_test'+kPara(2));        
    case 'exp'
        mat_kernel=(1+exp(kPara*mat_train*mat_test')).^(-1);
    otherwise 
        mat_kernel=mat_train*mat_test';
end