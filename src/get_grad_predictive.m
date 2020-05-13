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
%    get_grad obtains the gradient vector needed in Newton's method to
%    obtain the maximum of the approximate posterior.
%
%    INPUTS
%    f: the location at which the gradient is requestd (vector)
%    a: a vector containing the sum of all of the distances between the
%      survival times and the lower limit of their respective bins.
%    b: a vector containing the widths of each bin multiplied by the number
%      of observations which survive past the upper limit of the bin.
%    Sigmainv:  The prior precision matrix for the realized f.
%    ns: a vector containing the number of uncensored observations which
%      fall into each bin.
%
%    OUTPUT
%    out: the gradient vector at f.


function out = get_grad_predictive(f,ab1, ab2, ab3, ab4, ns1, ns2, ns3, ns4, C1, C2, C3, C4, Sigmainv)
    Sigmainv_f = Sigmainv * f;
    out = get_grad_predictive0(f, ab1, ns1, C1) + ...
        get_grad_predictive0(f, ab2, ns2, C2) + ...
        get_grad_predictive0(f, ab3, ns3, C3) + ...
        get_grad_predictive0(f, ab4, ns4, C4) + ...
        - Sigmainv_f;
end

