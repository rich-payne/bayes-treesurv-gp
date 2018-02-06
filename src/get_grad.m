function out = get_grad(f,a,b,Sigmainv,ns)
    Sigmainv_f = Sigmainv * f;
    out = ns - exp(f) .* (a + b) - Sigmainv_f;
end
