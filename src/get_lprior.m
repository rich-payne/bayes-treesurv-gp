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
%    get_lprior returns the log of the prior on tau and l
%
%    INPUTS
%    tau: the magnitude hyperparmeter
%    l: the length-scale hyperparameter
%
%    OUTPUTS
%    lprior: the log of the prior on tau and l
%    sigma2_mu: Not used. Ignore.

function [lprior,sigma2_mu] = get_lprior(tau,l)
    lprior = log(tpdf(tau ./ 10,1)) + ...   % proportional to a t-dist with variance of 10
         log(tpdf(l,1)); %+ ...
         %log(normpdf(mu,0,10));
     sigma2_mu = 100;
end