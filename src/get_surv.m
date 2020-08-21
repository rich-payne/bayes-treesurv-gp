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
%    Plots the posterior survival curve for a set of data.
%
%    INPUTS
%    Y_orig: A two-column matrix with the survival times in the first column
%      (on original scale) and the survival indicator in
%      the second column (0 if censored, 1 if observed).
%    res: the output from get_marginal function
%    ndraw:  The number of Monte Carlo draws to estimate credible regions
%    graph: If 1, a graph is produced. 0 suppresses graphing.
%    ystar: A grid where the posterior survival function will be evaluated.
%      If empty, grid is determined automatically.
%    alpha: the significance level for the credible itnervals.
%
%    OUTPUT
%    out: A structure with the following fields
%      surv: the ndraw posterior draws of the survival function
%      ystar: the grid on which the posterior survival function is
%        evaluated.  If the res input was created using the scaled data,
%        ystar will need to be re-scaled to plot on original units.
%      pmean: the posterior mean of the survival function
%      CI: the 1-alpha credible intervals
   

function out = get_surv(Y_orig,res,ndraw,graph,ystar,alpha)
    Ymax = max(Y_orig(:,1));
    if isempty(ystar)
        nstar = 100;
        ystar = linspace(.001,max(res.s)-1e-6,nstar); % Grid to evaluate the survival function
    else
        nstar = length(ystar);
    end
    binind = zeros(nstar,1);
    K = length(res.s) - 1;
    for ii=1:nstar
        [~,I] = max( ystar(ii) < res.s(2:(K+1)) );
        if I == 0 % ystar falls beyond the maximum s
            I = K; % put ystar value in the last bin for extrapolation...
        end
        binind(ii) = I;
    end
    thesurv = zeros(nstar,1);
    samps = chol(res.Omegainv) \ normrnd(0,1,length(res.f),ndraw);
    samps = (samps + res.f)';
    SURV = zeros(size(samps,1),nstar);
    for ii=1:size(samps,1)
        f = samps(ii,:)';
        for jj=1:nstar
            a = ystar(jj) - res.s(binind(jj));
            if binind(jj) > 1
                thediff = diff(res.s');
                b = sum( thediff(1:(binind(jj)-1)) .* exp(f(1:(binind(jj)-1))) );
            else
                b = 0;
            end
            thesurv(jj) = exp( - exp(f(binind(jj)))*a - b );
            %[a,b,binind(jj)]
            %f(1:(binind(jj)-1))'
        end
        %if(any(diff(thesurv) < 0))
            %thesurv
        %end
        SURV(ii,:) = thesurv;
    end
    out.surv = SURV;
    out.ystar = ystar;
    
    pmean = mean(SURV);
    qtiles = quantile(SURV,[alpha/2,1-alpha/2],1);

    out.pmean = pmean;
    out.CI = qtiles;
    if graph
        plot(ystar*Ymax,pmean,':k',ystar*Ymax,qtiles(1,:),'--k',ystar*Ymax,qtiles(2,:),'--k')
        graphpoints = 0;
        if graphpoints
            survpoints = interp1(ystar*Ymax,pmean,Y_orig);
            hold on;
                ind = Y_orig(:,2) == 1;
                Ysub = Y_orig(ind,1);
                plot(Ysub,survpoints(ind),'bo');
                Ysub = Y_orig(~ind,1);
                plot(Ysub,survpoints(~ind),'rx');
            hold off
        end
    end   
end