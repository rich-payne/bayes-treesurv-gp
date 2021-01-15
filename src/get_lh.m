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
%    Plots the posterior log hazard for a set of data.
%
%    INPUTS
%    Y_orig: A two-column matrix with the survival times in the first column
%      (on original scale) and the survival indicator in
%      the second column (0 if censored, 1 if observed).
%    res: the output from get_marginal function
%    ndraw:  The number of Monte Carlo draws to estimate credible regions
%    graph: If 1, a graph is produced. 0 suppresses graphing.
%    ystar: A grid where the posterior log hazard function will be evaluated.
%      If empty, grid is determined automatically.
%    alpha: the significance level for the credible itnervals.
%
%    OUTPUT
%    out_lh: A structure with the following fields
%      lhr: the ndraw posterior draws of the log hazard function
%      ystar: the grid on which the posterior log hazard function is
%        evaluated.  If the res input was created using the scaled data,
%        ystar will need to be re-scaled to plot on original units.
%      pmean: the posterior mean of the log hazard function
%      CI: the 1-alpha credible intervals

function out_lh = get_lh(Y_orig,res,varargin)
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
    LH = draw_lh(res, ystar, ndraw);
    if ~isempty(res2)
        LH2 = draw_lh(res2, ystar, ndraw);
        ind = binornd(1, p_trt_effect, size(LH, 1), 1);
        LH_FINAL = zeros(size(LH, 1), size(LH, 2));
        for ii = 1:length(ind)
            if ind == 0
                LH_FINAL(ii, :) = LH(ii, :);
            else
                LH_FINAL(ii, :) = LH2(ii, :);
            end
        end    
    else      
        LH_FINAL = LH; 
    end
    pmean = mean(LH_FINAL);
    qtiles = quantile(LH_FINAL, [alpha/2, 1 - alpha / 2], 1);
    out_lh.lhr = LH_FINAL;    
    out_lh.ystar = ystar;
    out_lh.pmean = pmean;
    out_lh.CI = qtiles;
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


function LHR = draw_lh(res, ystar, ndraw) 
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
    %thesurv = zeros(nstar,1);
    samps = chol(res.Omegainv) \ normrnd(0,1,length(res.f),ndraw);
    samps = (samps + res.f)';
    LHR = zeros(size(samps,1),nstar);
    for ii=1:size(samps,1)
        LHR(ii, :) = samps(ii, binind);
    end
end