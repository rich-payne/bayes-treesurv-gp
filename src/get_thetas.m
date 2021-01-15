%    bayes-treesurv-gp provides a Bayesian tree partition model to flexibly 
%    estimate survival functions in various regions of the covariate space.
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


function [tau,l] = get_thetas(ns,a,b,Z,tau,l,nugget,eps)
    options = optimset('Display','off');
    %lb = [1e-9,1e-9];
    lb = [0,0];
    ub = [];
    x0 = [tau,l];
    %warning off;
    out = fmincon(@(x) get_f_opt(ns,a,b,Z,x(1),x(2),nugget,eps),x0,[],[],[],[],lb,ub,[],options);
    %warning on;
    tau = out(1);
    l = out(2);
end

% Optimization function used in get_thetas
function neg_marg_y = get_f_opt(ns,a,b,Z,tau,l,nugget,eps)
    [~,marg_y] = get_f(ns,a,b,Z,tau,l,nugget,eps);
    %lprior = log(exppdf(1/tau^2,1)) + ... %encourages large taus
    %         log(exppdf(l^2,1)); % encourages small l (more flexible survival functions)   
%     lprior = log(tpdf(1 ./ tau ,1)) + ...   
%          log(tpdf(l,1)) + ...
%          log(normpdf(mu,0,10)); % enocourages small l
    %lprior = log(tpdf(tau ./ sqrt(10),1)) + ... %encourages large taus
    %  log(tpdf(l,1)); % encourages small l (more flexible survival functions)   
    lprior = get_lprior(tau,l);
    neg_marg_y = -(marg_y + lprior);
    %neg_marg_y = -marg_y;
end





