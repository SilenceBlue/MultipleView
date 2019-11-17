function [omega, extra] = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt,extrain)
% Construct the positive (semi-) definite and symmetric kernel matrix
% 
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
% 
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
% 
% >> Omega_Xt = kernel_matrix(X, kernel_fct, sig2, Xt)
% 
%
% Full syntax
% 
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
% >> Omega = kernel_matrix(X, kernel_fct, sig2, Xt)
% 
% Outputs    
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs    
%   X      : N x d matrix with the inputs of the training data
%   kernel : Kernel type (by default 'RBF_kernel')
%   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   Xt(*)  : Nt x d matrix with the inputs of the test data
% 
% See also:
%  RBF_kernel, lin_kernel, kpca, trainlssvm, kentropy


% Copyright (c) 2010,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.ac.be/sista/lssvmlab


gpuflag = gpuDeviceCount > 0 && license('test','distrib_computing_toolbox');
%gpuflag = false;
[nb_data,d] = size(Xtrain);

% if nb_data> 3000,
%   error(' Too memory intensive, the kernel matrix is restricted to size 3000 x 3000 ');
% end

%if size(Xtrain,1)<size(Xtrain,2),
%  warning('dimension of datapoints larger than number of datapoints?');
%end
stflag = 0;

if ~kernel_pars(1)
    stflag = 1;
end;


switch lower(kernel_type)
    case 'rbf_kernel'
        if length(kernel_pars)==d
            if size(kernel_pars,1)==d
                    kernel_pars=kernel_pars';
            end;
            
            nzidx = kernel_pars > 1e-6;
            kernel_pars = kernel_pars(nzidx);
            Xtrain = Xtrain(:,nzidx);
            tempN = Xtrain.^2;
            tempN = tempN./repmat(kernel_pars,nb_data,1);
            dd = sum(tempN,2);clear tempN;
            
            if nargin < 4                                
                omega = exp((2*(Xtrain*diag(1./kernel_pars)*Xtrain')-...
                      dd*ones(1,nb_data)-ones(nb_data,1)*dd'));
            else
                Xt = Xt(:,nzidx);
                ddtest = sum((Xt.^2)./repmat(kernel_pars,size(Xt,1),1),2);
                
                omega = exp((2*Xtrain*diag(1./kernel_pars)*Xt'-...
                    dd*ones(1,size(Xt,1))-ones(nb_data,1)*ddtest'))';
                
            end
            
        else
            if nargin < 4
                if ~gpuflag
                    dd = sum(Xtrain.^2,2);
                    onev = ones(nb_data,1);
                    omega = exp((2*(Xtrain*Xtrain') - dd*onev' -...
                        onev*dd')/kernel_pars(1));
                else
                    dd=gpuArray(sum(Xtrain.^2,2));
                    Xtrain = gpuArray(Xtrain);
                    omega=gather(exp((2*(Xtrain*Xtrain')-dd*gpuArray.ones(1,nb_data)-...
                        gpuArray.ones(nb_data,1)*dd')/kernel_pars(1)));
                    
                end;
            else
                if ~gpuflag
                    omega = exp((2*Xtrain*Xt'-sum(Xtrain.^2,2)*ones(1,size(Xt,1))-...
                        ones(nb_data,1)*sum(Xt.^2,2)')/kernel_pars(1))';
                else
                    
                    Xtrain = gpuArray(Xtrain);
                    Xt = gpuArray(Xt);
                    omega = gather(exp((2*Xtrain*Xt'-sum(Xtrain.^2,2)*gpuArray.ones(1,size(Xt,1))-...
                        gpuArray.ones(nb_data,1)*sum(Xt.^2,2)')/kernel_pars(1)))';
                end;                
            end

        end;
    case 'weightedrbf_kernel'
        
        if nargin<4
            omega = zeros(nb_data,nb_data);
            
            for i=1:size(Xtrain,1)
                for j=i:size(Xtrain,1)
                    omega(i,j)=exp(-(Xtrain(i,:)-Xtrain(j,:))*kernel_pars{2}*...
                        (Xtrain(i,:)-Xtrain(j,:))'/kernel_pars{1});
                    omega(j,i)=omega(i,j);
                end;
            end;
        else
            % test kernel matrix
            
        end;
         case 'cosinerbf_kernel'
        
        if nargin<4
          omega=exp(-squareform(pdist(Xtrain,'cosine').^2)/kernel_pars(1));

        else
          omega=exp(-(pdist2(Xt,Xtrain,'cosine').^2)/kernel_pars(1));

            
        end;
    case 'chisquared_kernel'
        if nargin<4,       
            omega=exp(-squareform(pdist(Xtrain,@chi2distance))/kernel_pars(1));
        else
%             omega=zeros(nb_data,size(Xt,1))';
%             for i=1:nb_data
%                 omega(:,i)=exp(-chi2distance(Xtrain(i,:),Xt)/kernel_pars(1));  
%             end;
%             omega=omega';
            omega=exp(-pdist2(Xt,Xtrain,@chi2distance)/kernel_pars(1));
        end;
        
    case 'lin_kernel'
        if nargin<4,
            omega = (Xtrain*Xtrain')';
        else
            omega = (Xtrain*Xt')';
        end
        
    case 'poly_kernel'
        if length(kernel_pars)>1
            t = kernel_pars(1); 
            d = kernel_pars(2); 
        else
            d = kernel_pars(1);
            t = 1; 
        end
     
        if nargin<4,
            omega=((Xtrain*Xtrain'+ t^2).^d)';
        else
            omega=((Xtrain*Xt'+ t^2).^d)';
        end;

  
    case 'normpoly_kernel'
        if length(kernel_pars)>1,
            t = kernel_pars(1); 
            d = kernel_pars(2); 
        else
            d = kernel_pars(1);
            t = 1; 
        end
        if nargin<4,
            D = sparse(diag(1./sqrt((sum(Xtrain.^2,2)+t^2).^d)));
            omega=(D*(Xtrain*Xtrain'+t^2).^d*D)';
        else
            D1 = sparse(diag(1./sqrt((sum(Xtrain.^2,2)+t^2).^d)));
            D2 = sparse(diag(1./sqrt((sum(Xt.^2,2)+t^2).^d)));
            omega = (D1 * (Xtrain*Xt'+t^2).^d * D2)';
        end;    
end


