function FS = fisherscore(data, labels)

% function FS = fisherscore(data, labels)
%
% Function to compute discriminant power of features with Fisher Score for 
% multi-class feature selection
% 
% Output: 
% FS: Vector with Fisher Score discriminability of features
%                 
% Inputs: 
% 
% data: Data matrix of size samples x features
% labels: Array of size samples with data labels

ulbl = unique(labels);
M1 = squeeze(mean(data(find(labels==ulbl(1)),:),1));
S1 = squeeze(std(data(find(labels==ulbl(1)),:),1));

M2 = squeeze(mean(data(find(labels==ulbl(2)),:),1));
S2 = squeeze(std(data(find(labels==ulbl(2)),:),1));

FS = abs(M1-M2)./sqrt(S1.^2 + S2.^2);
