function [tau,l] = get_thetas(ns,a,b,Z,tau,l,nugget,eps)
    options = optimset('Display','off');
    %lb = [1e-9,1e-9];
    lb = [0,0];
    ub = [];
    x0 = [tau,l];
    out = fmincon(@(x) get_f_opt(ns,a,b,Z,x(1),x(2),nugget,eps),x0,[],[],[],[],lb,ub,[],options);
    tau = out(1);
    l = out(2);
end

function neg_marg_y = get_f_opt(ns,a,b,Z,tau,l,nugget,eps)
    [~,marg_y] = get_f(ns,a,b,Z,tau,l,nugget,eps);
    lprior = log(exppdf(1/tau^2,1)) + ... %encourages large taus
             log(exppdf(l^2,1)); % encourages small l (away from prior mean)   
    neg_marg_y = -(marg_y + lprior);
end