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
%    get_marginal obtains the marginal of the data which is approximated
%      using Laplace approximations.
%
%    Y: a two-column matrix with the survival times in the first column
%      (scaled to be on the interval (0,1]) and the survival indicator in
%      the second column (0 if censored, 1 if observed).
%    K: the number of piecewise constant regions (bins) the hazard function 
%       will be estimated with.  Will be replaced if s is specified.
%    s: the bin boundaries of length K+1.  If s is empty, s is constructed
%      automatically using K.
%    eps: the threshold used to determine convergence of Newton's method.
%    tau: the starting value of the magnitude hyperparameter
%    l: the starting value of the length-scale hyperparameter
%    nugget: Not used. Ignore. Use 0.
%    EB: The empirical Bayes method to be used.  Currently, only one
%      optimization procedure is available.  1 optimizes hyperparameters,
%      0 leaves them at their provided values.
%
%    OUTPUT
%    marg_y: the approximate marginal of the data.
%    out: A structure with the following fields:
%      f: the MAP estimate of the latent process of the approximate posterior
%      marg_y: the approximate marginal of the data.
%      tau: the value of tau
%      l: the value of l
%      Omegainv: the precision matrix of the approximate posterior of f.
%      s: the bin boundaries of the hazard function
%      ns: a vector containing the number of uncensored observations which
%        fall into each bin.
%      a: a vector containing the sum of all of the distances between the
%        survival times and the lower limit of their respective bins.
%      b: a vector containing the widths of each bin multiplied by the number
%        of observations which survive past the upper limit of the bin.
%      ms: a vector containing the number of observations which survival
%        past each bin.
%      binind: an index specifying which bin each observation belongs to
%      Z: the bin centers
%      lprior: the value of the logged prior for tau and l

function [marg_y,out] = get_marginal(Y,K,s,eps,tau,l,nugget,EB)
    % Assumes Y has been scaled on the interval (0,1].
    %   but doesn't throw an error at the moment.
    %warning('off','MATLAB:nearlySingularMatrix')
    max_cntr_EM = 100;
    tol_EM = 1e-9;
    n = size(Y,1);
    if(isempty(s))
        %s = linspace(0,max(Y(:,1)) + min(.1,max(Y(:,1))/100),K+1);
        s = linspace(0,1.01,K+1);
    else
        K = length(s) - 1;
    end
    % An index of which of the K bins each observation falls in
    binind = zeros(n,1);
    for ii=1:n
        %Y(ii,1) < s(2:(K+1))
        [~,I] = max( Y(ii,1) < s(2:(K+1)) );
        binind(ii) = I;
    end
    
    % get rid of Ks above the largest one
    maxbin = max(binind);
    % reassign s & K
    s = s(1:(maxbin+1));
    K = maxbin;
    
    % Get rid of empty Ks
    % emptyind = [];
    %for ii=1:K
    %    if sum(binind == ii) == 0
    %        emptyind = [emptyind ii];
    %    end
    %end
    %s(emptyind) = [];
    % K = length(s) - 1;
    Z = (s(1:K) + diff(s)/2)';
    
    % Rescale data, Z, and s;
    % theta_hat = sum(Y(:,2)) / sum(Y(:,1)); % estimate of exponential survival function parameter
%     Z = Z .* theta_hat;
%     Y(:,1) = Y(:,1) .* theta_hat;
%     s = s .* theta_hat;
    
    % An index of which of the K bins each observation falls in
    % Recalculated with new s
    binind = zeros(n,1);
    for ii=1:n
        %Y(ii,1) < s(2:(K+1))
        [~,I] = max( Y(ii,1) < s(2:(K+1)) );
        binind(ii) = I;
    end
    
    % Number of uncensored observations in each bin
    yobs = Y(Y(:,2) == 1,1);
    ns = zeros(K,1);
    for ii=1:K
        ns(ii) = sum( (yobs > s(ii)) & (yobs <= s(ii+1)) );
    end
    
    ms = zeros(K,1); % Number of observations which survive past each bin
    for ii=1:K
        ms(ii) = sum(Y(:,1) > s(ii+1));
    end
    
    a = zeros(K,1);
    for ii=1:K
        a(ii) = sum( Y(binind == ii,1) - s(ii) );
    end
    b = diff(s') .* ms;    
            
    % Here's where EB happens...
    if EB == 1
        [tau,l] = get_thetas(ns,a,b,Z,tau,l,nugget,eps);
    elseif EB  == 2
        % [tau,l] = get_thetas_EM(Y,tau,l,nugget,K,max_cntr_EM,tol_EM);
    end
    
    %if justf
    %    f = get_f(ns,a,b,Z,tau,l,eps);
    %else
    [f,marg_y,Omegainv] = get_f(ns,a,b,Z,tau,l,nugget,eps);
    %end
    if ~isreal(marg_y) % Avoid complex numbers
        marg_y = -Inf;
    end
    
    if nargout > 1
        out.f = f;
        out.marg_y = marg_y;
        out.tau = tau;
        out.l = l;
        % out.mu = mu;
        out.Omegainv = Omegainv;
        out.s = s;
        out.a = a;
        out.b = b;
        out.ns = ns;
        out.ms = ms;
        out.binind = binind;
        out.Z = Z;
        out.lprior = get_lprior(tau,l); % prior on hyperparmeters...
        % out.theta_hat = theta_hat;
    end
end