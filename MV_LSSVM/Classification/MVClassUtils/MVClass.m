function model=MVClass(Xtrain,Ytrain,kernel,kernel_params,gamma,rho,assignment,Xtest)
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


%multi-view/class params
Nviews = length(Xtrain);
if ~iscell(Ytrain)
   ytr = Ytrain;
   Ytrain = cell(1,Nviews);
   for v=1:Nviews
       Ytrain{v} = ytr;
   end
end
if ~iscell(kernel)
    kerneltmp=kernel;
    kernel = cell(1,Nviews);
    for v=1:Nviews
        kernel{v}=kerneltmp;
    end
end
M = size(Ytrain{1},2);
if ~iscell(kernel_params)
    if isscalar(kernel_params)
       param = kernel_params;
       kernel_params = cell(1,Nviews);
       for v=1:Nviews
           kernel_params{v} = ones(1,M)*param;
       end
    elseif isempty(kernel_params) %lin_kernel
        kernel_params = cell(1,Nviews);
       for v=1:Nviews
           kernel_params{v} = ones(1,M)*0;
       end
    else
        param = kernel_params;
        kernel_params = cell(1,Nviews);
        for v=1:Nviews
           kernel_params{v} = param;
        end
    end
else
    if isscalar(kernel_params{1})
        for v=1:Nviews
            param = kernel_params{v};
            kernel_params{v} = ones(1,M)*param;
        end
    end
end
if ~iscell(gamma)
    if isscalar(gamma)
       gam = gamma;
       gamma = cell(1,Nviews);
       for v=1:Nviews
           gamma{v} = ones(1,M)*gam;
       end
    else
       gam = gamma;
       gamma = cell(1,Nviews);
       for v=1:Nviews
           gamma{v} = gam;
       end
    end
else
    if isscalar(gamma{1})
        for v=1:Nviews
            gam = gamma{v};
            gamma{v} = ones(1,M)*gam;
        end
    end
end

N = size(Xtrain{1},1);
model.Omegas = cell(1,Nviews);
model.yhat = cell(1,Nviews);
S = cell(1,M);
for l=1:M
    % Compute kernel matrices
    for v=1:Nviews
        model.Omegas{v}{l}=kernel_matrix(Xtrain{v},kernel{v},kernel_params{v}(l));
        Yh = Ytrain{v}(:,l)*Ytrain{v}(:,l)';
        model.Omegas{v}{l} = Yh.*model.Omegas{v}{l};
    end

    %Eigenvalue problem - calculate alpha, beta
    try
        L= zeros(Nviews*(N+1));
    catch
        L= sparse(zeros(Nviews*(N+1)));
    end
    for v1=1:Nviews
       L(v1,Nviews+1+(v1-1)*N: Nviews+v1*N) = Ytrain{v1}(:,l)';
       L(Nviews+1+(v1-1)*N: Nviews+v1*N,v1) = gamma{v1}(l)*Ytrain{v1}(:,l);
       L(Nviews+1+(v1-1)*N: Nviews+v1*N,Nviews+1+(v1-1)*N: Nviews+v1*N) = (gamma{v1}(l)*model.Omegas{v1}{l}+eye(N));
       for v2=1:Nviews
           if v1~=v2
               L(Nviews+1+(v1-1)*N: Nviews+v1*N,v2) = rho*Ytrain{v2}(:,l);
               L(Nviews+1+(v1-1)*N: Nviews+v1*N,Nviews+1+(v2-1)*N: Nviews+v2*N) = rho*model.Omegas{v2}{l};
           end
       end           
    end
    %model.L{l} = L;
    try
        R = zeros(Nviews*(N+1),1);
    catch
        R = sparse(zeros(Nviews*(N+1),1));
    end
          
    for v=1:Nviews
        R(Nviews+1+(v-1)*N: Nviews+v*N) = gamma{v}(l) + rho*(Nviews-1);
    end
    try
        ba= L \ R;
    catch
        L=sparse(L);
        R=sparse(R);
        ba = L\R;
    end
    for v=1:Nviews
        model.b{v}(l) = real(ba(v));
        model.alphas{v}(:,l) = real(ba(Nviews+1+(v-1)*N: Nviews+v*N));
    end

    model.ba{l} = ba;
    
    %Classifier
    for v=1:Nviews
        model.yhat{v}(:,l) = model.Omegas{v}{l}*model.alphas{v}(:,l) + model.b{v}(l)*ones(N,1);
    end
    
    if ischar(assignment)
        switch lower(assignment)
            case 'uncoupled'
                for v=1:Nviews
                    model.yhat{v}(:,l) = sign(model.yhat{v}(:,l));
                end
            case 'mean'
                yhat_total = zeros(N,1);
                for v=1:Nviews
                    yhat_total =  yhat_total + (1/Nviews) * model.yhat{v}(:,l);
                end
                for v=1:Nviews
                    model.yhat{v}(:,l) = sign(yhat_total);
                end
            case 'median'
                yhat_total = zeros(Nviews,N);
                for v=1:Nviews
                    yhat_total(v,:) = model.yhat{v}(:,l)'; 
                end
                for v=1:Nviews
                    model.yhat{v}(:,l) = sign(median(yhat_total)');
                end
            case 'committee'
                % Determine the error covariance matrix C
                c = cell(1,Nviews);
                for v=1:Nviews
                    c{v} = abs(model.yhat{v}(:,l) - Ytrain{v}(:,l));
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
                
                %Determine classification
                model.yhat_total(:,l) = zeros(N,1);
                for v=1:Nviews
                    model.yhat_total(:,l) =  model.yhat_total(:,l) + beta(v) * model.yhat{v}(:,l);
                end
                for v=1:Nviews
                    model.yhat{v}(:,l) = sign(model.yhat_total(:,l));
                end
            otherwise
                error('assignment should have the value: "uncoupled", "mean", "median", "committee" or an array of weights ');
        end
    else
        if length(assignment)~= Nviews
            error('assignment should have the value: "uncoupled", "mean", "median", "committee" or an array of weights ');
        end
        model.yhat_total(:,l) = zeros(N,1);
        for v=1:Nviews
            model.yhat_total(:,l) =  model.yhat_total(:,l) + assignment(v) * model.yhat{v}(:,l);
        end
        for v=1:Nviews
            model.yhat{v}(:,l) = sign(model.yhat_total(:,l));
        end
    end
end

if exist('Xtest','var') && ~isempty(Xtest)
    
    Nt = size(Xtest{1},1);
    model.Omegatest = cell(1,Nviews);
    model.yhat_test = cell(1,Nviews);
    for l=1:M
        % Compute kernel matrix
        for v=1:Nviews
            model.Omegatest{v}{l}=kernel_matrix(Xtrain{v},kernel{v},kernel_params{v}(l),Xtest{v});
            Yh = ones(Nt,1)*Ytrain{v}(:,l)';
            model.Omegatest{v}{l} = Yh.*model.Omegatest{v}{l};
        end

        %Classifier
        for v=1:Nviews
            model.yhat_test{v}(:,l) = model.Omegatest{v}{l}*model.alphas{v}(:,l) + model.b{v}(l)*ones(Nt,1);
        end
        
        if ischar(assignment)
            switch lower(assignment)
                case 'uncoupled'
                    for v=1:Nviews
                        model.yhat_test{v}(:,l) = sign(model.yhat_test{v}(:,l));
                    end
                case 'mean'
                    yhat_ttotal = zeros(Nt,1);
                    for v=1:Nviews
                        yhat_ttotal =  yhat_ttotal + (1/Nviews) * model.yhat_test{v}(:,l);
                    end
                    for v=1:Nviews
                        model.yhat_test{v}(:,l) = sign(yhat_ttotal);
                    end
                case 'median'
                    yhat_ttotal = zeros(Nviews,Nt);
                    for v=1:Nviews
                        yhat_ttotal(v,:) = model.yhat_test{v}(:,l)'; 
                    end
                    for v=1:Nviews
                        model.yhat_test{v}(:,l) = sign(median(yhat_ttotal)');
                    end
                case 'committee'
                    model.yhat_ttotal(:,l) = zeros(Nt,1);
                    for v=1:Nviews
                        model.yhat_ttotal(:,l) =  model.yhat_ttotal(:,l) + beta(v) * model.yhat_test{v}(:,l);
                    end
                    for v=1:Nviews
                        model.yhat_test{v}(:,l) = sign(model.yhat_ttotal(:,l));
                    end
                otherwise
                    error('assignment should have the value: "uncoupled", "mean", "committee" or an array of weights ');
            end
        else
            if length(assignment)~= Nviews
                error('assignment should have the value: "uncoupled", "mean", "committee" or an array of weights ');
            end
            model.yhat_ttotal(:,l) = zeros(Nt,1);
            for v=1:Nviews
                model.yhat_ttotal(:,l) =  model.yhat_ttotal(:,l) + assignment(v) * model.yhat_test{v}(:,l);
            end
            for v=1:Nviews
                model.yhat_test{v}(:,l) = sign(model.yhat_ttotal(:,l));
            end
        end
    end
end

