function out = ldet(A)
    L = chol(A);
    out = 2*sum(log(diag(L)));
end