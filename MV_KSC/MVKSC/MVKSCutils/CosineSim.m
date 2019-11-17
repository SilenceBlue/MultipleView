function [collinearity] = CosineSim(projval,clustermembership)

k = size(projval,2) + 1;
N = size(projval,1);

%Mean Calculation
clustermean = zeros(k,size(projval,2));
for i=1:k
    tempdata = projval((clustermembership(:)==i),:);
    clustermean(i,:) = mean(tempdata);
end;
collinearity = zeros(k,1);
clusterinfo = zeros(N,2);

%Re-Calculate Membership
normclustermean = sqrt(sum(clustermean.^2,2));
for l = 1:N
   %max = -inf;
   %for i=1:k
   %    cosineval = (1.0*dot(clustermean(i,:),projval(l,:)))/(norm(clustermean(i,:))*norm(projval(l,:)));
   %    if (cosineval>max)
   %        max = cosineval;
   %        index = i;
   %    end
   %end
   normmean = normclustermean*norm(projval(l,:));
   cosineval = clustermean*projval(l,:)'./normmean;
   [max_value,index] = max(cosineval);
   clusterinfo(l,:) = [max_value index];
end;

%Calculating Cosine Similarity for entire cluster
for i=1:k
    indexes = find(clusterinfo(:,2)==i);
    if (~isempty(indexes))
        collinearity(i) = 1.0*mean(clusterinfo(indexes,1));
    else
        collinearity(i)= 0.0;
    end;
end;

collinearity(isnan(collinearity),:) = 0;
collinearity(isinf(collinearity),:) = 0;
