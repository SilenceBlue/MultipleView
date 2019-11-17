function par=aveRBFPar(data,size)
mat_temp=sum(data.^2,2)*ones(1,size)...
    +ones(size,1)*sum(data.^2,2)'...
    -2*data*data';

tempMean=1/size^2*sum(sum(mat_temp,1),2);
par=sqrt(tempMean);