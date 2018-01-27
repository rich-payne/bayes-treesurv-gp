function Sigma = get_sigma(Z,tau,l,nugget)
    K = length(Z);
    if K > 1
        Sigma = tau^2 * exp(-(1/(2*l^2)) * squareform(pdist(Z,'squaredeuclidean')) ) + diag(ones(1,K)*(nugget.^2));
    else
        Sigma = tau^2 + nugget^2;
    end
end
    