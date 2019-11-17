function model=MVRegr(Xtrain,Ytrain,kernel,kernel_params,gamma,rho,couplingMatrix,assignment,Xtest)
% 
%     Inputs: Xtrain:  Nviews x 1 cell matrix of N x d_v matrices of training data
%             Ytrain:  N x m matrix of labels with m output. Alternatively
%                      a Nviews x 1 cell matrix for each view.
%             kernel:  kernel type (e.g., 'RBF_kernel'), can be
%                      view-specific by giving a Nviews x 1 cell matrix
%             kernel_params:  two options:
%                       - kernel parameters (e.g., sig2) for all views and classes the same
%                       - Nviews x 1 cell matrix with kernel parameters for
%                       each view seperately
%             gamma:   regularisation parameter(s). Can be scalar (same
%                      gamma for each view and class) or Nviews x 1 array (different
%                      gamma for each view and class)
%             rho: Coupling parameter
%             couplingMatrix:
%                       - 'eye': Identity matrix
%                       - 'dd': based on the kernel degree matrices
%             assignment: class assigment manner: 
%                       - 'uncoupled': class assignment is done
%                       separately for view
%                       - 'mean': class assignment is done on mean all
%                       e-values
%                       - 'median': class assignment is done on median all
%                       e-values
%                       - Nviews x 1 array:  class assignment is done on 
%                       e_total, which is calculated with the given array 
%                       of weights
%                       - 'committee' class assignment is done on
%                       e_total, which is calculated according to committee
%                       network principle
%          (*) Xtest:  Nviews x 1 cell matrix of Nt x d_v matrices of test data
tic;
N = size(Xtrain{1},1);
%multi-view params
Nviews = length(Xtrain); 
if isscalar(kernel_params)
   param = kernel_params;
   kernel_params = cell(1,Nviews);
   for v=1:Nviews
       kernel_params{v} = param;
   end
end
if isscalar(gamma)
   gam = gamma;
   gamma = cell(1,Nviews);
   for v=1:Nviews
       gamma{v} = gam;
   end
end
if ~iscell(Ytrain)
   ytr = Ytrain;
   Ytrain = cell(1,Nviews);
   for v=1:Nviews
       Ytrain{v} = ytr;
   end
end


% Compute kernel matrix
model.Omegas = cell(1,Nviews);
for v=1:Nviews
    model.Omegas{v}=kernel_matrix(Xtrain{v},kernel,kernel_params{v});
end

%Define S
S = cell(Nviews,Nviews);
if ischar(couplingMatrix)
    switch lower(couplingMatrix)
        case 'eye'
            for v1=1:Nviews
                for v2=1:Nviews
                    S{v1,v2} = diag(eye(N));
                end
            end
        case 'dd'
            Dinv = cell(1,Nviews);
            for v=1:Nviews
                d= sum(model.Omegas{v},2);
                dinv_tmp = 1./d;
                Dinv{v} = sparse(diag(dinv_tmp));
            end
            for v1=1:Nviews
                for v2=1:Nviews
                    S{v1,v2} = diag(sqrt(Dinv{v1})*sqrt(Dinv{v2}));
                end
            end
        otherwise
            error('couplingMatrix should have the value: "eye", "DD", or a V x V cell of N x 1 matrices diagonal vectors');
    end
else
    if size(S,1) ~= Nviews || size(S,2) ~= Nviews || ~iscell(S)
        error('couplingMatrix should have the value: "eye", "DD", or a V x V cell of N x 1 matrices diagonal vectors');
    else
        S = couplingMatrix;
    end
end


%Eigenvalue problem - calculate alpha, beta
L= zeros(Nviews*(N+1));
for v1=1:Nviews
   L(v1,Nviews+1+(v1-1)*N: Nviews+v1*N) = ones(1,N);
   L(Nviews+1+(v1-1)*N: Nviews+v1*N,v1) = gamma{v1}*ones(N,1);
   L(Nviews+1+(v1-1)*N: Nviews+v1*N,Nviews+1+(v1-1)*N: Nviews+v1*N) = (gamma{v1}*model.Omegas{v1}+eye(N));
   for v2=1:Nviews
       if v1~=v2
           L(Nviews+1+(v1-1)*N: Nviews+v1*N,v2) = sparse(rho*S{v1,v2}.*ones(N,1));
           L(Nviews+1+(v1-1)*N: Nviews+v1*N,Nviews+1+(v2-1)*N: Nviews+v2*N) = rho*diag(S{v1,v2})*model.Omegas{v2};
       end
   end           
end
R = zeros(Nviews*(N+1),1);
SsumY = cell(1,Nviews);
for v1=1:Nviews
    SsumY{v1} = zeros(N,1);
    for v2=1:Nviews
        if v1~=v2
            SsumY{v1} = SsumY{v1} + S{v1,v2}.*Ytrain{v2};
        end
    end
end            
for v=1:Nviews
    R(Nviews+1+(v-1)*N: Nviews+v*N) = gamma{v}*Ytrain{v} + rho*SsumY{v};
end

ba= L \ R;
for v=1:Nviews
    model.b{v} = real(ba(v));
    model.alphas{v} = real(ba(Nviews+1+(v-1)*N: Nviews+v*N));
end

%Regressor
model.yhat = cell(1,Nviews);
for v=1:Nviews
    model.yhat{v} = model.Omegas{v}*model.alphas{v} + model.b{v}*ones(N,1); 
end

if ischar(assignment)
    switch lower(assignment)
        case 'uncoupled'
            %do nothing
        case 'mean'
            model.yhat_total = zeros(N,1);
            for v=1:Nviews
                model.yhat_total =  model.yhat_total + (1/Nviews) * model.yhat{v};
            end
            for v=1:Nviews
                model.yhat{v} = model.yhat_total;
            end
        case 'median'
            model.yhat_total = zeros(Nviews,N);
            for v=1:Nviews
                model.yhat_total(v,:) = model.yhat{v}'; 
            end
            for v=1:Nviews
                model.yhat{v} = median(model.yhat_total)';
            end
        case 'committee'
            % Determine the error covariance matrix C
            c = cell(1,Nviews);
            for v=1:Nviews
                c{v} = abs(model.yhat{v} - Ytrain{v});
            end
            Cov = zeros(Nviews);
            for v=1:Nviews
                for v2=1:Nviews
                   Cov(v,v2) = mean(c{v}.*c{v2});
                end
            end
            Cinv = inv(Cov);
            sum_Cinv = sum(sum(Cinv));

            %Determine the weights beta
            beta = zeros(Nviews,1);
            for v=1:Nviews
               for v2=1:Nviews
                   beta(v) = beta(v) + Cinv(v,v2);
               end
               beta(v) = beta(v)/sum_Cinv;
            end

            %Determine regression
            model.yhat_total = zeros(N,1);
            for v=1:Nviews
                model.yhat_total =  model.yhat_total + beta(v) * model.yhat{v};
            end
            for v=1:Nviews
                model.yhat{v} = model.yhat_total;
            end
        otherwise
            error('assignment should have the value: "uncoupled", "mean", "median", "committee" or an array of weights ');
    end
else
    if length(assignment)~= Nviews
        error('assignment should have the value: "uncoupled", "mean", "median", "committee" or an array of weights ');
    end
    model.yhat_total = zeros(N,1);
    for v=1:Nviews
        model.yhat_total =  model.yhat_total + assignment(v) * model.yhat{v};
    end
    for v=1:Nviews
        model.yhat{v} = model.yhat_total;
    end
end

if exist('Xtest','var') && ~isempty(Xtest)
    
    Nt = size(Xtest{1},1);
    
    % Compute kernel matrix
    model.Omegatest = cell(1,Nviews);
    for v=1:Nviews
        model.Omegatest{v}=kernel_matrix(Xtrain{v},kernel,kernel_params{v},Xtest{v});
    end
    
    %Regressor
    for v=1:Nviews
        model.yhat_test{v} = model.Omegatest{v}*model.alphas{v} + model.b{v}*ones(Nt,1); 
    end
    
    if ischar(assignment)
        switch lower(assignment)
            case 'uncoupled'
                %do nothing
            case 'mean'
                model.yhat_ttotal = zeros(Nt,1);
                for v=1:Nviews
                    model.yhat_ttotal =  model.yhat_ttotal + (1/Nviews) * model.yhat_test{v};
                end
                for v=1:Nviews
                    model.yhat_test{v} = model.yhat_ttotal;
                end
            case 'median'
                model.yhat_ttotal = zeros(Nviews,Nt);
                for v=1:Nviews
                    model.yhat_ttotal(v,:) = model.yhat_test{v}'; 
                end
                for v=1:Nviews
                    model.yhat_test{v} = median(model.yhat_ttotal)';
                end
            case 'committee'
                model.yhat_ttotal = zeros(Nt,1);
                for v=1:Nviews
                    model.yhat_ttotal =  model.yhat_ttotal + beta(v) * model.yhat_test{v};
                end
                for v=1:Nviews
                    model.yhat_test{v} = model.yhat_ttotal;
                end
            otherwise
                error('assignment should have the value: "uncoupled", "mean", "median", "committee" or an array of weights ');
        end
    else
        if length(assignment)~= Nviews
            error('assignment should have the value: "uncoupled", "mean", "committee", "median" or an array of weights ');
        end
        model.yhat_ttotal = zeros(Nt,1);
        for v=1:Nviews
            model.yhat_ttotal =  model.yhat_ttotal + assignment(v) * model.yhat_test{v};
        end
        for v=1:Nviews
            model.yhat_test{v} = model.yhat_ttotal;
        end
    end
end
model.time = toc;

