function out = get_hess(f,a,b,Sigmainv,hessval)
    %tmp = -.5*Sigmainv;
    out = diag(-exp(f) .* (a + b)) - Sigmainv + ...
        + hessval;
end