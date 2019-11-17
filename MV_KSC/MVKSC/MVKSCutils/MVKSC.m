function model=MVKSC(Xtrain,kernel,params,k,gamma,assignment,Xtest)
% 
%     Multi-View Kernel Spectral Clustering [1]
%     
%     Inputs: Xtrain:  Nviews x 1 cell matrix of N x d_v matrices of training data
%             kernel:  kernel type (e.g., 'RBF_kernel')
%             params:  two options:
%                       - kernel parameters (e.g., sig2) for all views the same
%                       - Nviews x 1 cell matrix with kernel parameters for
%                       each view seperately
%             k:       number of desired clusters
%             gamma:   regularisation parameter(s). Can be scalar (same
%                      gamma for each view) or Nviews x 1 array (different
%                      gamma for each view)
%             assignment: cluster assigment manner: 
%                       - 'uncoupled': cluster assignment is done
%                       separately for view
%                       - 'mean': cluster assignment is done on mean all
%                       e-values
%                       - 'median': cluster assignment is done on mean all
%                       e-values
%                       - Nviews x 1 array:  cluster assignment is done on 
%                       e_total, which is calculated with the given array 
%                       of weights
%                       - 'committee' cluster assignment is done on
%                       e_total, which is calculated according to committee
%                       network principle
%          (*) Xtest:  Nviews x 1 cell matrix of Nt x d_v matrices of test data
%             
%     Output: 'model' structure containing:
%                     model.Omegas:       Nviews x 1 cell matrix of N x N kernel matrices.
%               model.OmegasCentered:     Nviews x 1 cell matrix of N x N centered kernel matrices.
%                     model.dinv:         Nviews x 1 cell matrix of N x 1 vectors containing the 
%                                         inverses of the node degrees for training data points.
%                     model.alpha:        Nviews x 1 cell matrix of N x (k-1) matrices containing the 
%                                         eigenvectors corresponding to the largest k-1 eigenvalues.
%                     model.rho:          (k-1) x 1 vector with the largest k-1 eigenvalues.
%                     model.etrain:       Nviews x 1 cell matrix of N x (k-1) matrices with the 
%                                         training data projections.
%          (*)        model.etrainTotal:  N x (k-1) matrix with the training data projections in 
%                                         case of coupled cluster assignment.
%                     model.C:            Nviews x 1 cell matrix of k x (k-1) matrices containing the 
%                                         cluster codewords.
%                     model.qtrain:       Nviews x 1 cell matrix of N x 1 vectors of cluster membership 
%                                         for training data.
%                     model.alphaCenters: Nviews x 1 cell matrix of k x (k-1) matrices with the cluster 
%                                         prototypes in the eigenvector representation.
%                     model.Cextra:       Nviews x 1 cell matrix of (k-1) x 1 cells with the cluster 
%                                         codebooks from 2 to k clusters.
%               model.alphaCentersExtra:  Nviews x 1 cell matrix of(k-1) x 1 cells with the cluster 
%                                         prototypes from 2 to k clusters.
%                     model.qtrainExtra:  Nviews x 1 cell matrix of N x (k-1) matrices with cluster 
%                                         memberships from 2 to k clusters.
%          (*)        model.Omegatest:    Nviews x 1 cell matrix of Nt x N kernel matrices of test points.
%          (*)  model.OmegastestCentered: Nviews x 1 cell matrix of Nt x N centered kernel matrices of 
%                                         test points.
%          (*)        model.dinvt:        Nviews x 1 cell matrix of Nt x 1 vectors containing the 
%                                         inverses of the node degrees for test data points.
%          (*)        model.etest:        Nviews x 1 cell matrix of Nt x (k-1) matrix with test data 
%                                         projections.
%          (*)        model.etestTotal:   Nt x (k-1) matrix with the test data projections in 
%                                         case of coupled cluster assignment.
%          (*)        model.zerocluster:  Nviews x 1 cell matrix of Nt x 1 vector with logical values 
%                                         indicating membership to the zero cluster.
%          (*)        model.alphat:       Nviews x 1 cell matrix of Nt x (k-1) matrix of out-of-sample 
%                                         eigenvectors.
%          (*)        model.qtest:        Nviews x 1 cell matrix of Nt x 1 vector of cluster membership 
%                                         for test data.
%          (*)        model.qtestExtra:   Nviews x 1 cell matrix of Nt x (k-1) matrix with test data cluster 
%                                         membership from 2 to k clusters. 
%                     model.time          Runtime of the entire method.
%  ------------------------------------------------------------------------------------------------------------
%
% (*) Optional
%           
%  [1] 
%
%  Copyright (c) 2012, Carlos Alzate, ESAT/SISTA, K.U.Leuven.              
                   

%% checks

tic;

Nviews = size(Xtrain,2);
if isscalar(gamma)
    gamma_tmp = gamma;
    gamma = ones(Nviews,1);
    gamma = gamma*gamma_tmp;
end
if ~iscell(params)
    params_tmp = params;
    params = cell(1,Nviews);
    for v=1:Nviews
        params{v} = params_tmp;
    end
end
        
model.Omegas = cell(1,Nviews); 

for v=1:Nviews
    Omegas_tmp=kernel_matrix(Xtrain{v},kernel,params{v});
    model.Omegas{v}=0.5*(Omegas_tmp+Omegas_tmp');
end

N = size(model.Omegas{1},1);
MINDEG_THR = 1e-3*N; 

%% Compute the kernel matrix Omega, and the degree matrix D

d = cell(1,Nviews);
Dinv = cell(1,Nviews);
for v=1:Nviews
    d{v}= sum(model.Omegas{v},2);
    model.dinv{v} = 1./d{v};
    Dinv{v} = sparse(diag(model.dinv{v}));
end

%% Compute the centered kernel matrices
model.OmegasCentered = cell(1,Nviews);
model.MD = cell(1,Nviews);
model.KD = cell(1,Nviews);
for v=1:Nviews
    model.MD{v} = eye(N)-ones(N,1)*model.dinv{v}'/sum(model.dinv{v});
    model.KD{v} = eye(N) - Dinv{v}*ones(N,1)*ones(1,N)/sum(model.dinv{v});
    model.OmegasCentered{v} = model.MD{v} * model.Omegas{v} * model.KD{v};
end


%% Compute the eigenvalue problem to get the alphas (and the bias term)
    
DOI = cell(1,Nviews);
for v=1:Nviews
   DOI{v} = eye(N) - (gamma(v)/N)*Dinv{v}*model.OmegasCentered{v};
end

L= zeros(Nviews*N);
col=1;
for v=1:Nviews
    for v2=1:Nviews
        if v2~=v
            L((v-1)*N+1:v*N,col:col+N-1) = sqrt(Dinv{v})*sqrt(Dinv{v2})*model.OmegasCentered{v2};
        end
        col = col+N;
    end
    col=1;
end
R= zeros(Nviews*N);
col=1;
for v=1:Nviews
    R((v-1)*N+1:v*N,col:col+N-1) = DOI{v};
    col = col+N;
end

try
    [alphas,rhoInv]=eigs(L,R,(k-1),'lm'); 
catch
    [alphas,rhoInv]=eig(L,R);
    [~,i]=sort(rhoInv,'descend');clear a;
    rhoInv=rhoInv(i(1:k-1));
    alphas=alphas(:,i(1:k-1));
end

for v=1:Nviews
    model.alpha{v}=real(alphas((v-1)*N+1:v*N,:)); 
end
rhoInv=real(diag(rhoInv));
model.rho = 1./rhoInv;


%% Compute the score variables for training data;

model.etrain = cell(1,Nviews);
for v=1:Nviews
    model.etrain{v} = model.OmegasCentered{v}*model.alpha{v};
end

%Compute the e-values used for training depending on the type of cluster
%assigment
etrain_used = cell(1,Nviews);
if ischar(assignment)
    switch lower(assignment)
        case 'uncoupled'
            etrain_used = model.etrain;
        case 'mean'
            model.etrainTotal = zeros(N,(k-1));
            for v=1:Nviews
                model.etrainTotal = model.etrainTotal + (1/Nviews) * model.etrain{v};
            end
            for v=1:Nviews
                etrain_used{v} = model.etrainTotal;
            end
        case 'median'
            model.etrainTotal = zeros(N,(k-1));
            for j=1:k-1
                etrainTotal_tmp = zeros(Nviews,N);
                for v=1:Nviews
                    etrainTotal_tmp(v,:) = model.etrain{v}(:,j);
                end
                model.etrainTotal(:,j) = median(etrainTotal_tmp)';
            end
            for v=1:Nviews
                etrain_used{v} = model.etrainTotal;
            end
        case 'committee'
            qtrainTmp = cell(1,Nviews);
            for v=1:Nviews
                 [~,qtrainTmp_tmp,~] = ...
                     KSCcodebook(model.etrain{v},model.alpha{v});
                 qtrainTmp{v} = qtrainTmp_tmp;
            end

            % Determine the error covariance matrix S
            s = cell(1,Nviews);
            for v=1:Nviews
                s_tmp = silhouette(Xtrain{v},qtrainTmp{v});
                s_tmp = (s_tmp+1)/2; %bring to [0;1]
                s{v} = 1-s_tmp; %error
            end
            S = zeros(Nviews);
            for v=1:Nviews
                for v2=1:Nviews
                   S(v,v2) = mean(s{v}.*s{v2});
                end
            end
            Sinv = inv(S);
            sum_Sinv = sum(sum(Sinv));

            %Determine the weights beta
            model.beta = zeros(Nviews,1);
            for v=1:Nviews
               for v2=1:Nviews
                   model.beta(v) = model.beta(v) + Sinv(v,v2);
               end
               model.beta(v) = model.beta(v)/sum_Sinv;
            end

            %Determine etrainTotal
            model.etrainTotal = zeros(N,(k-1));
            for v=1:Nviews
                model.etrainTotal = model.etrainTotal + model.beta(v) * model.etrain{v};
            end
            for v=1:Nviews
                etrain_used{v} = model.etrainTotal;
            end
        otherwise
            error('assignment should have the value: "uncoupled", "mean", "committee" or an array of weights ');
    end
else
    if length(assignment)~= Nviews
        error('assignment should have the value: "uncoupled", "mean", "committee" or an array of weights ');
    end
    model.etrainTotal = zeros(N,(k-1));
    for v=1:Nviews
        model.etrainTotal = model.etrainTotal + assigment(v) * model.etrain{v};
    end
    for v=1:Nviews
        etrain_used{v} = model.etrainTotal;
    end
end

 model.C = cell(1,Nviews);
 model.qtrain = cell(1,Nviews);
 model.alphaCenters = cell(1,Nviews);
 for v=1:Nviews
     [model.C{v},model.qtrain{v},model.alphaCenters{v}] = ...
         KSCcodebook(etrain_used{v},model.alpha{v});
 end

model.Cextra = cell(1,Nviews);
model.alphaCentersExtra = cell(1,Nviews);
model.qtrainExtra = cell(1,Nviews);
for v=1:Nviews
    Cextra = cell(k-1,1);
    alphaCentersExtra = cell(k-1,1);
    qtrainExtra = zeros(N,k-1);
    for j=2:k
       [Cextra{j-1},qtrainExtra(:,j-1),...
            alphaCentersExtra{j-1}] = ...
                   KSCcodebook(etrain_used{v}(:,1:j-1),model.alpha{v}(:,1:j-1));
    end; 
    model.Cextra{v} = Cextra;
    model.alphaCentersExtra{v} = alphaCentersExtra;
    model.qtrainExtra{v} = qtrainExtra;
end 

if exist('Xtest','var') && ~isempty(Xtest)
    
    Nt = size(Xtest{1},1);
    
    for v=1:Nviews
        model.Omegastest{v}=kernel_matrix(Xtrain{v},kernel,params{v},Xtest{v});
    end
    
    %% Size of Omegatest should ALWAYS be [Nt x N]
    
    model.zeroclusters = cell(1,Nviews);
    dt = cell(1,Nviews);
    model.dinvt = cell(1,Nviews);
    for v=1:Nviews
        model.zeroclusters{v} = sum(model.Omegastest{v},2)<=MINDEG_THR;
        dt{v} = sum(model.Omegastest{v},2);
        model.dinvt{v}=1./dt{v};   
        model.dinvt{v}(model.zeroclusters{v}) = 0;
    end
           
    %% Compute the centered kernel matrices
    model.OmegastestCentered = cell(1,Nviews);
    MDt = cell(1,Nviews);
    for v=1:Nviews
        MDt{v} = eye(Nt)-ones(Nt,1)*model.dinvt{v}'/sum(model.dinvt{v});
        model.OmegastestCentered{v} = MDt{v} * model.Omegastest{v} * model.KD{v};
    end
   
    model.etest = cell(1,Nviews);
    for v=1:Nviews
        model.etest{v} = model.OmegastestCentered{v}*model.alpha{v};  
    end
    
    %Compute the e-values used for testing depending on the type of cluster
    %assigment
    etest_used = cell(1,Nviews);
    if ischar(assignment)
        switch lower(assignment)
            case 'uncoupled'
                etest_used = model.etest;
            case 'mean'
                model.etestTotal = zeros(Nt,(k-1));
                for v=1:Nviews
                    model.etestTotal = model.etestTotal + (1/Nviews) * model.etest{v};
                end
                for v=1:Nviews
                    etest_used{v} = model.etestTotal;
                end
            case 'median'
            model.etestTotal = zeros(Nt,(k-1));
            for j=1:k-1
                etestTotal_tmp = zeros(Nviews,Nt);
                for v=1:Nviews
                    etestTotal_tmp(v,:) = model.etest{v}(:,j);
                end
                model.etestTotal(:,j) = median(etestTotal_tmp)';
            end
            for v=1:Nviews
                etest_used{v} = model.etestTotal;
            end
            case 'committee'
                %Determine etrainTotal
                model.etestTotal = zeros(Nt,(k-1));
                for v=1:Nviews
                    model.etestTotal = model.etestTotal + model.beta(v) * model.etest{v};
                end
                for v=1:Nviews
                    etest_used{v} = model.etestTotal;
                end
        end
    else
        model.etestTotal = zeros(Nt,(k-1));
        for v=1:Nviews
            model.etestTotal = model.etestTotal + assigment(v) * model.etest{v};
        end
        for v=1:Nviews
            etest_used{v} = model.etestTotal;
        end
    end
    
    model.alphat = cell(1,Nviews);
    for v=1:Nviews
        sum_e = zeros(Nt,k-1);
        for v2=1:Nviews
            if v~=v2
                sum_e = sum_e + ((sqrt(model.dinvt{v}).*sqrt(model.dinvt{v2}))*ones(1,k-1)).*model.etest{v2};
            end
        end
        
        for j=1:k-1
            model.alphat{v}(:,j) = gamma(v)/N * model.etest{v}(:,j).*model.dinvt{v} + model.rho(j)*sum_e(:,j);
        end;
    end

    for v=1:Nviews
        model.qtest{v} = KSCmembership(etest_used{v},model.C{v},...
            model.alphat{v},model.alphaCenters{v});
    end
    
    %Model selectection criteria BLF & Cosine similarity
    model.qtestExtra = cell(1,Nviews);
    model.collinearity = cell(1,Nviews);
    model.balance = cell(1,Nviews);
    for v=1:Nviews
        for j=2:k
            %BLF
            model.qtestExtra{v}(:,j-1)=KSCmembership(etest_used{v}(:,1:j-1),...
                model.Cextra{v}{j-1},model.alphat{v}(:,1:j-1),model.alphaCentersExtra{v}{j-1});

            [model.collinearity{v}{j-1},model.balance{v}(j-1)]=BLF(etest_used{v}(:,1:j-1),...
                model.qtestExtra{v}(:,j-1),dt{v});
        end
    end
    %Cosine similarity
    for v=1:Nviews
        model.CosineSim(v) = mean(CosineSim(etest_used{v},model.qtest{v}));
    end
   
end;

model.time=toc;
