function out = get_grad(f,a,b,mu,Sigmainv,ns)
    val = -.5 * Sigmainv * f - .5 * diag(Sigmainv) .* f;   
    out = ns - exp(f) .* (a + b) + val + sum(mu .* Sigmainv)';
end
