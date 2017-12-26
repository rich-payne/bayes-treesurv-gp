function [tau,l,mu] = get_thetas(ns,a,b,mu,Z,tau,l,nugget,eps)
    options = optimset('Display','off');
    %lb = [1e-9,1e-9];
    lb = [0,0,-Inf];
    ub = [];
    x0 = [tau,l,mu];
    %warning off;
    out = fmincon(@(x) get_f_opt(ns,a,b,x(3),Z,x(1),x(2),nugget,eps),x0,[],[],[],[],lb,ub,[],options);
    %warning on;
    tau = out(1);
    l = out(2);
    mu = out(3);
end

function neg_marg_y = get_f_opt(ns,a,b,mu,Z,tau,l,nugget,eps)
    [~,marg_y] = get_f(ns,a,b,mu,Z,tau,l,nugget,eps);
    %lprior = log(exppdf(1/tau^2,1)) + ... %encourages large taus
    %         log(exppdf(l^2,1)); % encourages small l (more flexible survival functions)   
    lprior = log(tpdf(tau/sqrt(10),1)) + ...   
         log(tpdf(l,1)) + ...
         log(normpdf(mu,0,10)); % enocourages small l
    %lprior = log(tpdf(tau ./ sqrt(10),1)) + ... %encourages large taus
    %  log(tpdf(l,1)); % encourages small l (more flexible survival functions)   


    neg_marg_y = -(marg_y + lprior);
end