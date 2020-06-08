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

function [f_final,marg_y,Omegainv] =  get_f_predictive(ns1, ns2, ns3, ns4, ab1, ab2, ab3, ab4, Z1, Z2, Z3, Z4,tau1, tau2, tau3, tau4,l1, l2, l3, l4,eps)
    % warning('off','MATLAB:nearlySingularMatrix')
    [Sigmainv1, fhat1] = get_sigma_f(Z1, tau1, l1, ns1, ab1);
    [Sigmainv2, fhat2] = get_sigma_f(Z2, tau2, l2, ns2, ab2);
    [Sigmainv3, fhat3] = get_sigma_f(Z3, tau3, l3, ns3, ab3);
    [Sigmainv4, fhat4] = get_sigma_f(Z4, tau4, l4, ns4, ab4);
    
    % Initial value for Newton's method
    f = [fhat1; fhat2; fhat3; fhat4];
    Sigmainv = blkdiag(Sigmainv1, Sigmainv2, Sigmainv3, Sigmainv4);
        
    if(~all(isfinite(f)))
        error('fhat is not finite');
    end
    
    ok = 0;
    cntr = 0;
    cntr_max = 100; 
    while(~ok)
       thehess = get_hess_predictive(f, ab1, ab2, ab3, ab4, ns1, ns2, ns3, ns4, C1, C2, C3, C4, Sigmainv);
       thegrad = get_grad_predictive(f, ab1, ab2, ab3, ab4, ns1, ns2, ns3, ns4, C1, C2, C3, C4, Sigmainv);
       fnew = f - (thehess \ thegrad);
       if (sum(abs(fnew - f)) < eps) || (cntr == cntr_max)
           ok = 1;
           f_final = fnew;
       end
       if cntr == cntr_max
           disp('Maximum number of iterations reached in Newton''s method');
       end
       f = fnew;
       cntr = cntr + 1;
    end
    if nargout == 1
        return;
    else
        Sigmainv_f = Sigmainv * f_final;
        g0val = get_g0(f, ab1, ns1, C1) + ...
            get_g0(f2, ab2, ns2, C2) + ...
            get_g0(f3, ab3, ns3, C3) + ...
            get_g0(f4, ab4, ns4, C4) - ...
            .5 * f_final' * Sigmainv_f;
        Omegainv = -get_hess_predictive(f, ab1, ab2, ab3, ab4, ns1, ns2, ns3, ns4, C1, C2, C3, C4, Sigmainv);
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

function out = get_g0(f, ab, ns, C)
    kq = length(ns);
    out = ones(1, kq) * (diag(ns) * C * f - exp(C * f) .* ab);
    out = out';
end


function [Sigmainv, fhat] = get_sigma_f(Z, tau, l, ns, ab)
    if(size(Z,1) < size(Z,2))
        error('Z must be a column vector');
    end    
    K = length(Z);
    % Get ML estimate (from simple case..., not actual MLE) to start newton's algorithm
    fhat = log(ns) - log(ab);
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
    if K > 1
        dz = Z(2) - Z(1);
    else
        dz = [];
    end
    Sigmainv = get_sigma_inv(K,tau,l,dz);
end
