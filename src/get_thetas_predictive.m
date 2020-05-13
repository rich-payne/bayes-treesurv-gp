%    bayes-treesurv-gp provides a Bayesian tree partition model to flexibly 
%    estimate survival functions in various regions of the covariate space.
%    Copyright (C) 2017-2018  Richard D. Payne
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    This function chooses tau and l to maximize the approximate marginal
%    likelihood.
%
%    INPUTS
%    ns: a vector containing the number of uncensored observations which
%      fall into each bin.
%    a: a vector containing the sum of all of the distances between the
%      survival times and the lower limit of their respective bins.
%    b: a vector containing the widths of each bin multiplied by the number
%      of observations which survive past the upper limit of the bin.
%    Z: the bin centers
%    tau: the starting value of tau
%    l: starting value for l
%    nugget: Not used.  Use 0.
%    eps: convergence criteria for Newton's method.
%
%    OUTPUTS
%    tau: the optimized value of tau
%    l: the optimized value of l


function [tau,l] = get_thetas_predictive(ns1, ns2, ns3, ns4, ab1, ab2, ab3, ab4,Z1, Z2, Z3, Z4,tau1, tau2, tau3, tau4, l1, l2, l3, l4,eps)
    options = optimset('Display','off');
    %lb = [1e-9,1e-9];
    lb = zeros(1, 8);
    ub = [];
    x0 = [tau1, tau2, tau3, tau4, l1, l2, l3, l4];
    %warning off;
    out = fmincon(@(x) get_f_opt_predictive(ns1, ns2, ns3, ns4, ab1, ab2, ab3, ab4,Z1, Z2, Z3, Z4, x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), eps),x0,[],[],[],[],lb,ub,[],options);
    %warning on;
    tau = out(1:4);
    l = out(5:8);
end

% Optimization function used in get_thetas
function neg_marg_y = get_f_opt_predictive(ns1, ns2, ns3, ns4, ab1, ab2, ab3, ab4,Z1, Z2, Z3, Z4, tau1, tau2, tau3, tau4, l1, l2, l3, l4, eps)
    [~,marg_y] = get_f_predictive(ns1, ns2, ns3, ns4, ab1, ab2, ab3, ab4,Z1, Z2, Z3, Z4,tau1, tau2, tau3, tau4, l1, l2, l3, l4, eps);
    %lprior = log(exppdf(1/tau^2,1)) + ... %encourages large taus
    %         log(exppdf(l^2,1)); % encourages small l (more flexible survival functions)   
%     lprior = log(tpdf(1 ./ tau ,1)) + ...   
%          log(tpdf(l,1)) + ...
%          log(normpdf(mu,0,10)); % enocourages small l
    %lprior = log(tpdf(tau ./ sqrt(10),1)) + ... %encourages large taus
    %  log(tpdf(l,1)); % encourages small l (more flexible survival functions)   
    lprior = get_lprior(tau1,l1) + ...
        get_lprior(tau2,l2) + ...
        get_lprior(tau3,l3) + ...
        get_lprior(tau4,l4);
    
    neg_marg_y = -(marg_y + lprior);
    %neg_marg_y = -marg_y;
end





