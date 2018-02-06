function out = get_grad(f,a,b,Sigmainv,ns,gradval)
    Sigmainv_f = Sigmainv * f;
    val = gradval .* sum(Sigmainv_f) ;   
    out = ns - exp(f) .* (a + b) - Sigmainv_f + val;
end
