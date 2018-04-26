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
%    This function creates a plot analogous to a Mallow CP plot.  
%
%    INPUTS
%    varargin: output from Tree_Surv.  If the MCMC was done in pieces (e.g.
%      using the 'resume' argument), the outputs from each individual run
%      can be placed as an input, separated by commas.
%
%    OUTPUTS
%    trees: returns the tree with the highest marginal likelhood for each
%      tree size
%    LLIKES: the log-likelihoods for the best tree of each size
%    whichout: an index on varargin which specifies which input had the
%      highest likelihood for a given tree size
%    trees_lpost, LPOST, whichout_lpost: same as above, but the log-posterior
%       is used instead of the log-likelihood

function [trees,LLIKES,whichout,trees_lpost,LPOST,whichout_lpost] = mallow(varargin) 
    % submit output files for a mallows plot
    maxtree = 0;
    nout = length(varargin);
    for ii=1:nout
        output = varargin{ii};
        maxtree = max(max(output.treesize),maxtree);
    end
        
        
    % Mallow's C_p type plot
    % maxtree = max(output.treesize);
    LLIKES = [];
    LPOST = [];
    trees = cell(maxtree,1);
    trees_lpost = cell(maxtree,1);
    whichout = [];
    whichout_lpost = [];
    for ii=1:maxtree
        tmpll_old = -Inf;
        tmp_lpost_old = -Inf;
        LLIKES(ii) = NaN;
        LPOST(ii) = NaN;
        for jj=1:nout
            output = varargin{jj};
            ind = output.treesize == ii;
            llikes = output.llike(ind);
            lprior = output.lprior(ind);
            TREES = output.Trees(ind);
            [tmpll,I] = max(llikes);
            [tmp_lpost,I_lpost] = max(llikes + lprior);
            
            % Save LLIKE info
            if ~isempty(tmpll)
                if tmpll > tmpll_old
                    LLIKES(ii) = tmpll;
                    trees{ii} = TREES{I};
                    whichout(ii) = jj;
                    tmpll_old = tmpll;
                end
            end
%             else
%                 LLIKES(ii) = NaN;
%             end
            % LOG POSTERIOR INFO
            if ~isempty(tmp_lpost)
                if tmp_lpost > tmp_lpost_old
                    LPOST(ii) = tmp_lpost;
                    trees_lpost{ii} = TREES{I_lpost};
                    whichout_lpost(ii) = jj;
                    tmp_lpost_old = tmp_lpost;
                end
            end
        end
    end
    subplot(1,2,1)
    plot(1:maxtree,LLIKES,'-o')
    title('Log-Likelihood');
    xlabel('Tree Size')
    ylabel('Best p(y|T)')
    subplot(1,2,2)
    plot(1:maxtree,LPOST,'-o')
    title('Log-Posterior')
    xlabel('Tree Size')
    ylabel('Best p(y|T)p(T)')
end