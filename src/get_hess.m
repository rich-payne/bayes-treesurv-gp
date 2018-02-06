function out = get_hess(f,a,b,Sigmainv)
    ds = -exp(f) .* (a + b) - diag(Sigmainv);
    out = spdiags(ds,0,-Sigmainv); % Should be a sparse matrix object;
end