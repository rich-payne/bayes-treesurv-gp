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
%    along with this program.  If not, see <http://www.gnu.org/licenses/>
%
%    get_hess obtains the Hessian needed in Newton's method to
%    obtain the maximum of the approximate posterior.
%
%    INPUTS
%    f: the location at which the Hessian is requestd (vector)
%    a: a vector containing the sum of all of the distances between the
%      survival times and the lower limit of their respective bins.
%    b: a vector containing the widths of each bin multiplied by the number
%      of observations which survive past the upper limit of the bin.
%    Sigmainv:  The prior precision matrix for the realized f.
%
%    OUTPUT
%    out: the Hessian matrix at f.

function out = get_hess(f,a,b,Sigmainv)
    %ds = -exp(f) .* (a + b) - diag(Sigmainv);
    %out = spdiags(ds,0,-Sigmainv); % Should be a sparse matrix object;
    % this is much faster than spdiags
    ds2 = -exp(f) .* (a + b);
    out = -Sigmainv + diag(ds2);
end