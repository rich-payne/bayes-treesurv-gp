function [f_final,marg_y,Omegainv] =  get_f(ns,a,b,Z,tau,l,nugget,eps)
    %warning('off','MATLAB:nearlySingularMatrix')
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
    
    %Sigma_exp = ;
    Sigma = get_sigma(Z,tau,l,sqrt(nugget));
    %Sigmainv = 1/(tau^2) * inv(Sigma_exp);
    Sigmainv = inv(Sigma);
    
    
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
    [~,sigma2_mu] = get_lprior(tau,l);
    Astar = sum(sum(Sigmainv)) + 1 / sigma2_mu;
    hessval = 1./Astar .* (sum(Sigmainv,2) * sum(Sigmainv,1)); % quantity needed for Hessian
    gradval = 1./Astar .* sum(Sigmainv,2); % Quantity needed for gradient
    while(~ok)
       thehess = get_hess(f,a,b,Sigmainv,hessval);
       thegrad = get_grad(f,a,b,Sigmainv,ns,gradval);
       fnew = f - thehess \ thegrad;
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
        Sigmainv_f = Sigma \ f_final;
        g0val = sum(ns .* f_final) - sum(exp(f_final) .* (a + b)) - ...
            .5* f_final' * Sigmainv_f + ...
            .5 * (sum(Sigmainv_f).^2) ./ Astar;
        Omegainv = -get_hess(f_final,a,b,Sigmainv,hessval);
        try
            det1 = -.5*ldet(Omegainv,'chol');
        catch
            % det1 = -.5*ldet(Omegainv);
            marg_y = -Inf;
            return;
        end
        try
            det2 = -.5*ldet(Sigma,'chol');
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
        marg_y = det1 + det2 + g0val - .5*log(Astar);
    end
end