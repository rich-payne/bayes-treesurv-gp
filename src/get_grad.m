function out = get_grad(f,a,b,Astar,Sigmainv,ns)
    val = -.5 * Sigmainv * f - .5 * diag(Sigmainv) .* f + ...
        1/Astar .* sum(Sigmainv,2) .* sum(Sigmainv * f) ;   
    out = ns - exp(f) .* (a + b) + val;
end
