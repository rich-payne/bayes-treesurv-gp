function out = get_hess(f,a,b,Sigmainv)
    %ds = -exp(f) .* (a + b) - diag(Sigmainv);
    %out = spdiags(ds,0,-Sigmainv); % Should be a sparse matrix object;
    % this is much faster than spdiags
    ds2 = -exp(f) .* (a + b);
    out = -Sigmainv + diag(ds2);
end