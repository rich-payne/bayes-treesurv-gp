function out = get_grad(f,a,b,Astar,Sigmainv,ns)
    val = 1/Astar .* sum(Sigmainv,2) .* sum(Sigmainv * f) ;   
    out = ns - exp(f) .* (a + b) - Sigmainv * f + val;
end
