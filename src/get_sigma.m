function Sigma = get_sigma(Z,tau,l,nugget)
    K = length(Z);
    Sigma = tau^2 * exp(-(1/(2*l^2)) * squareform(pdist(Z,'squaredeuclidean')) ) + diag(ones(1,K)*(nugget.^2));
end
    