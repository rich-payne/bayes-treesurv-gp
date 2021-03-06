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
%    INPUTS
%    ns: a vector containing the number of uncensored observations which
%      fall into each bin.
%    a: a vector containing the sum of all of the distances between the
%      survival times and the lower limit of their respective bins.
%    b: a vector containing the widths of each bin multiplied by the number
%      of observations which survive past the upper limit of the bin.
%    Z: a vector containing the centers of the bins
%    tau: the magnitude hyperparameter for the covariance function of the
%      latent process, f.
%    l: the length-scale hyperparameter for the covariance function of the
%      latent process, f. 
%    nugget: unused argument.  Ignore.
%    eps: the threshold used to assess convergence in Newton's method.
%
%    OUTPUTS
%    f_final: the MAP estimate of the latent process, f.
%    marg_y: the approximated marginal of the data given tau, l.
%    Omegainv: the precision matrix for the latent process, f, utilizing
%      the Laplace approximation.  That is, p(f|y) is approximated (via the
%      Laplace approximation) to be N(f_final, Omegainv^{-1})

function [f_final,marg_y,Omegainv] =  get_f(ns,a,b,Z,tau,l,nugget,eps)
    % warning('off','MATLAB:nearlySingularMatrix')
    if(size(Z,1) < size(Z,2))
        error('Z must be a column vector');
    end
    K = length(Z);
    % Get ML estimate to start newton's algorithm
    fhat = log(ns) - log(a + b);
    ind = isfinite(fhat);
    %sum(ind)
    if ~all(ind)
        if sum(isfinite(fhat)) < 2
            fhat = normrnd(0,.01,K,1);
        else
            f_interp = interp1(Z(ind),fhat(ind),Z(~ind),'linear','extrap');
            fhat(~ind) = f_interp;
        end
    end
    %fhat = ones(K,1) .* hazardhat;
    
    %Sigma_exp = ;
    %Sigma = get_sigma(Z,tau,l,sqrt(nugget));
    %Sigmainv = 1/(tau^2) * inv(Sigma_exp);
    %Sigmainv = inv(Sigma);
    if K > 1
        dz = Z(2) - Z(1);
    else
        dz = [];
    end
    Sigmainv = get_sigma_inv(K,tau,l,dz);
    
    % Sigmainv * Sigma
    
    %inv(full(Sigmainv))
    %Sigma
    
    
    f = fhat; % Initial value for Newton's method
    
    
    
    
    if(~all(isfinite(fhat)))
        a
        b
        ns
        fhat
        error('fhat is not finite');
    end
    
    ok = 0;
    cntr = 0;
    cntr_max = 100; 
    %[~,sigma2_mu] = get_lprior(tau,l);
    %Astar = sum(sum(Sigmainv)) + 1 / sigma2_mu;
    %hessval = 1./Astar .* (sum(Sigmainv,2) * sum(Sigmainv,1)); % quantity needed for Hessian
    %gradval = 1./Astar .* sum(Sigmainv,2); % Quantity needed for gradient
    while(~ok)
       thehess = get_hess(f,a,b,Sigmainv);
       thegrad = get_grad(f,a,b,Sigmainv,ns);
       fnew = f - (thehess \ thegrad);
       if (sum(abs(fnew - f)) < eps) || (cntr == cntr_max)
           ok = 1;
           f_final = fnew;
       end
       if cntr == cntr_max
           % disp('Maximum number of iterations reached in Newton''s method');
       end
       f = fnew;
       cntr = cntr + 1;
    end
    if nargout == 1
        return;
    else
        Sigmainv_f = Sigmainv * f_final;
        g0val = sum(ns .* f_final) - sum(exp(f_final) .* (a + b)) - ...
            .5* f_final' * Sigmainv_f;
        Omegainv = -get_hess(f_final,a,b,Sigmainv);
        try
            det1 = -.5*ldet(Omegainv,'chol');
        catch
            % det1 = -.5*ldet(Omegainv);
            marg_y = -Inf;
            return;
        end
        try
            det2 = .5*ldet(Sigmainv,'chol');
        catch
            %det2 = -.5*ldet(Sigma);
            marg_y = -Inf;
            return;
        end
        %isreal(det1)
        %isreal(det2)
        %isreal(g0val)
        %lprior = get_lprior(tau,l,mu);
        % marg_y = det1 + det2 + g0val + lprior - .5*log(Astar);
        marg_y = det1 + det2 + g0val;
    end
end