function R2 = rsquared(data, labels)

% function R2 = rsquared(data, labels)
%
% Function to compute discriminant power of features with r^2 for 
% multi-class feature selection.
% 
% Output: 
% R2: Vector with r^2 discriminability of features
%                 
% Inputs: 
% 
% data: Data matrix of size samples x features
% labels: Array of size samples with data labels

R2 = diag(corr(data,repmat(labels,[1 size(data,2)]),'rows','complete')).^2;
