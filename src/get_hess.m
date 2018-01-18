function out = get_hess(f,a,b,Astar,Sigmainv)
    tmp = -.5*Sigmainv;
    out = diag(-exp(f) .* (a + b) - diag(Sigmainv),0) + tmp - diag(diag(tmp)) + ...
        + 1./Astar .* (sum(Sigmainv,2) * sum(Sigmainv,1));
end