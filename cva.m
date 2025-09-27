function CVA = cva(data, labels)

% function CVA = cva(data,labels)
%
% Function to compute discriminant power of features with Canonical Variate 
% Analysis (CVA) -- otherwise known as Multiple Discriminant Analysis-- for 
% multi-class feature selection.
% 
% Output: 
% CVA: Vector of size samples with discriminability of features
%                 
% Inputs: 
% 
% data: Data matrix of size samples x features
% labels: Array of size samples with data labels

% Force labels to be in [1,N]
UL = unique(labels);
copylabels = labels;
for i=1:length(UL)
    labels(copylabels==UL(i))=i;
end

NFeat=size(data,2);
NC=length(UL);

% Within-class and between-class scatter matrices
C = zeros(NFeat);
B = zeros(NFeat);
Ma = mean(data,1);
for c=1:NC
    IsClass = (labels==c);
    NClass = sum(IsClass);
    Mc = mean(data(IsClass,:));
    C = C + NClass*cov(data(IsClass,:));
    B = B + NClass*((Mc-Ma)'*(Mc-Ma));
end

% Generalized eigenvalue decomposition
[V,E] = eigs(pinv(C)*B,NC-1);
V = V(:,1:NC-1);
E = diag(E(1:NC-1,1:NC-1))';

% Data projection
Pdata = data*V;

% Calculate correlations between orignal features and maximum-discriminancy
% projections, weighted by corresponding eigenvalues
DP = sum(repmat(E,[NFeat 1]).* (corr(data,Pdata).^2),2);
DP = DP/sum(DP);

% Move back to matrix format
CVA = DP;
