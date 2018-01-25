function [lprior,sigma2_mu] = get_lprior(tau,l)
    lprior = log(tpdf(tau ./ 10,1)) + ...   % proportional to a t-dist with variance of 10
         log(tpdf(l,1)); %+ ...
         %log(normpdf(mu,0,10));
     sigma2_mu = 100;
end