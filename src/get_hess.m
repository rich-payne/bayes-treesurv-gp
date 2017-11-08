function out = get_hess(f,a,b,Sigmainv)
    tmp = -.5*Sigmainv;
    out = diag(-exp(f) .* (a + b) - diag(Sigmainv),0) + tmp - diag(diag(tmp));
end