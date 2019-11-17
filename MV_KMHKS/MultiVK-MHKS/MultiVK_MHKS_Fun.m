function  subResult=MultiVK_MHKS_Fun(classOne,classTwo,inPutInf,FM)
% The basic MultiVK-MHKS function

sizeClassOne=size(classOne,1);
sizeClassTwo=size(classTwo,1);
totalSize=sizeClassOne+sizeClassTwo;
label=[ones(sizeClassOne,1);-1*ones(sizeClassTwo,1)];

% Generate Orignal Kernel Matrix
kName=inPutInf.kernel;
kPar=inPutInf.kPar;

% Generate Multi-View Kernel Matrix
M=inPutInf.M;
K_multi=cell(M,1);
T=cell(M,1);
w=cell(M,1);

for p=1:M;
    K_multi{p}=[GeKernel(classOne,classOne,kName{p},kPar{p}), -1*GeKernel(classOne,classTwo,kName{p},kPar{p});...
        -1*GeKernel(classTwo,classOne,kName{p},kPar{p}), GeKernel(classTwo,classTwo,kName{p},kPar{p})];
    T{p}=ones(totalSize,1)/sum(ones(totalSize,1));
    w{p}=1;
end;

tempKernelAlignment=0;
index=0;
for i=1:M;
    for j=i+1:M;
        index=index+1;
        tempKernelAlignment=tempKernelAlignment+KernelAlignment(K_multi{i},K_multi{j});
    end;
end;
subResult.KA=tempKernelAlignment/index;

% Algorithm Information
C=inPutInf.C;
u=inPutInf.u;
gamma=inPutInf.gamma;
R=inPutInf.R;
B=ones(totalSize,1)*inPutInf.B';
maxIter=inPutInf.sizeIter;
iter=1;
termi=inPutInf.termination;
e=cell(M,1);

J=loss(K_multi,T,w,label,B,C,gamma,totalSize,u,M,FM);
for p=1:M;
    AA=(1+gamma)*FM*K_multi{p}+C(p)*eye(totalSize);
    BB=(1+gamma)*FM*label;
    DD=(1+gamma)*label'*FM*K_multi{p};
    EE=(1+gamma)*label'*FM*label;
    pin{p}=pinv([AA,BB;DD,EE]);
end;
while iter<maxIter;
    for p=1:M;
        CC=FM*(1+B(:,p)+gamma*sumMultiView(K_multi,T,w,label,M,u));
        FF=label'*FM*(1+B(:,p)+gamma*sumMultiView(K_multi,T,w,label,M,u));
        X=pin{p}*[CC;FF];
        T{p}=X(1:totalSize);
        w{p}=X(totalSize+1);        
        
        e{p}=K_multi{p}*T{p}+w{p}*label-ones(totalSize,1)-B(:,p);
        B(:,p)=B(:,p)+R(p)*(e{p}+abs(e{p}));
    end;
    
    J_before=J;
    matJ(iter)=J;
    J=loss(K_multi,T,w,label,B,C,gamma,totalSize,u,M,FM); 
    deltJ=abs(J-J_before);
    if deltJ/J_before<=termi;
        break;
    end;    
    iter=iter+1;    
end;
subResult.T=T;
subResult.w=w;