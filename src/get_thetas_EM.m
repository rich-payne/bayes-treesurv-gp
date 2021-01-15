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
%    THIS FUNCTION IS NOT YET OPERATIONAL.


function [tau,l] = get_thetas_EM(Y,tau,l,nugget,K,max_cntr,tol)
    warning('off','MATLAB:nearlySingularMatrix')
    options = optimset('Display','off');
    % Get initial f for given values of tau and l
    [~,res] = get_marginal(Y,K,[],eps,tau,l,nugget,0);
    f = res.f;
    lb = [0,0];
    ub = [];
    ok = 0;
    cntr = 0;
    while ~ok
       cntr = cntr + 1;
       % Maximize the hyperpamaters 
       out = fmincon(@(x) opt_EM(f,res.Z,x(1),x(2),res.Omegainv,nugget),[tau,l],[],[],[],[],lb,ub,[],options);
       tau_new = out(1);
       l_new = out(2);
       if (abs(tau_new - tau) + abs(l_new - l)) < tol
           ok = 1;
       elseif cntr == max_cntr
           ok = 1;
           disp('Maximum iterations reached for EM optimization');
       end
       tau = tau_new;
       l = l_new;
       % Draw a new f
       [~,res] = get_marginal(Y,K,[],eps,tau,l,nugget,0);
       f = res.f;
    end
    warning('off','MATLAB:nearlySingularMatrix')
end

function out = opt_EM(f,Z,tau,l,Omegainv,nugget)
% [tau,l]
    K = length(Z);
    Sigma = tau^2 * exp(-(1/(2*l^2)) * squareform(pdist(Z,'squaredeuclidean')) ) + diag(ones(1,K)*nugget);
    Sigmainv = inv(Sigma);
    %Omegainv = -get_hess(f,a,b,Sigmainv);
    tmp1 = -.5 * (trace(Omegainv \ Sigmainv) + f' * (Sigma \ f));
    tmp2 = -.5 * ldet(Sigma,'chol');
    out = -(tmp1 + tmp2);
end