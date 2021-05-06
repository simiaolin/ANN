function [RMS] = mypca(X,X_mean,V,D,q)
% mypca Summary of this function goes here 
% compresses the entire dataset by projecting it onto q principal components, then reconstructs it and
% measures the reconstruction error.

X_stan = X - X_mean;

eigenValues = diag(D);
[eigenValues,ind] = sort(eigenValues,"descend");

% create the projection matrix ET
ET = V(:,ind(1:q));
ET = ET';

% compress the dataset
X_reduced = ET * (X_stan)';
X_reduced = X_reduced';
% reconstruct the dataset 
X_reconstructed = X_reduced * ET;
X_reconstructed = X_reconstructed + X_mean;

% measure the reconstruction error
RMS = (sqrt(mean(mean((X_reconstructed-X).^2))));

end

