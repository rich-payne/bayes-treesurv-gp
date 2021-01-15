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
   

function out = get_surv(Y_orig,res,varargin)
    p = inputParser;
    addRequired(p, 'Y_orig');
    addRequired(p, 'res');
    addParameter(p, 'ndraw', 10000);
    addParameter(p, 'graph', 1);
    addParameter(p, 'ystar', []);
    addParameter(p, 'alpha', .05);
    addParameter(p, 'res2', []);
    addParameter(p, 'p_trt_effect', []);
    parse(p, Y_orig, res, varargin{:});
    R = p.Results;
    Y_orig = R.Y_orig;
    res = R.res;
    ndraw = R.ndraw;
    graph = R.graph;
    ystar = R.ystar;
    alpha = R.alpha;
    res2 = R.res2;
    p_trt_effect = R.p_trt_effect;
    
    Ymax = max(Y_orig(:,1));
    if isempty(ystar)
        nstar = 100;
        ystar = linspace(.001,max(res.s)-1e-6,nstar); % Grid to evaluate the survival function
    else
        nstar = length(ystar);
    end
    SURV = draw_surv(res, ystar, ndraw);
    if ~isempty(res2)
        SURV2 = draw_surv(res2, ystar, ndraw);
        ind = binornd(1, p_trt_effect, size(SURV, 1), 1);
        SURV_FINAL = zeros(size(SURV, 1), size(SURV, 2));
        for ii = 1:length(ind)
            if ind == 0
                SURV_FINAL(ii, :) = SURV(ii, :);
            else
                SURV_FINAL(ii, :) = SURV2(ii, :);
            end
        end    
    else      
        SURV_FINAL = SURV; 
    end
    out.surv = SURV_FINAL;  
    out.ystar = ystar;
    
    pmean = mean(SURV_FINAL);
    qtiles = quantile(SURV_FINAL,[alpha/2,1-alpha/2],1);

    out.pmean = pmean;
    out.CI = qtiles;
    if graph
        plot(ystar*Ymax,pmean,':k',ystar*Ymax,qtiles(1,:),'--k',ystar*Ymax,qtiles(2,:),'--k')
        hold on;
            %plot(ystar*Ymax,mean(SURV), ':r');
            %plot(ystar*Ymax,mean(SURV2), ':b');
        hold off;
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

function SURV = draw_surv(res, ystar, ndraw) 
    nstar = length(ystar);
    binind = zeros(nstar,1);
    K = length(res.s) - 1;
    for ii=1:nstar
        [Imax,I] = max( ystar(ii) < res.s(2:(K+1)) );
        if Imax == 0 % ystar falls beyond the maximum s
            I = K; % put ystar value in the last bin for extrapolation...
        end
        binind(ii) = I;
    end
    thesurv = zeros(nstar,1);
    samps = chol(res.Omegainv) \ normrnd(0,1,length(res.f),ndraw);
    samps = (samps + res.f)';
    SURV = zeros(size(samps,1),nstar);
    thediff = diff(res.s');
    for ii=1:size(samps,1)
        f = samps(ii,:)';
        for jj=1:nstar
            a = ystar(jj) - res.s(binind(jj));
            if binind(jj) > 1
                b = sum( thediff(1:(binind(jj)-1)) .* exp(f(1:(binind(jj)-1))) );
            else
                b = 0;
            end
            thesurv(jj) = exp( - exp(f(binind(jj)))*a - b );
        end
        
        SURV(ii,:) = thesurv;
    end
    
end