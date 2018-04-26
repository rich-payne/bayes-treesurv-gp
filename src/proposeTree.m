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
%    Proposes tree for the reversible jump MCMC algorithm with parallel
%      tempering.  Specifically, this is used in Tree_Surv().
%
%    INPUTS
%    T: Current tree in MCMC chain
%    y: response variable and censoring indicator (2 columns)
%    X: covariate matrix
%    allprobs: probability of a grow, prune, change, and swap step (vector of
%           length 4)
%     p: probability of moving to the next largest or smallest value for
%       continuous rules in a change step rather than draw a rule from a prior.
%       temp: inverse temperature of the tree
%     mset: 1 if proposing tree for a multiset, 0 otherwise.

%     OUTPUTS
%     Tstar: proposed tree
%     prop_ratio: the logged proposal ratio of the MH algorithm
%     r: integer (1-4) indicating whether a grow, prune, change, or swap step
%       was seleted, respectively.
%     lr: the log of the MH ratio
function [Tstar,prop_ratio,r,lr] = proposeTree(T,y,X,allprobs,p,temp,mset)
    Tprior = T.Prior;
    % Initial values
    lr = [];
    % See what moves are possible
    if length(T.Allnodes) >= 5
        [~,swappossible] = swap(T,y,X,[],0);
    else
        swappossible = [];
    end
    [nbirths,T] = nbirthnodes(T,X);
    [p_g,p_p,p_c,p_s] = propprobs(T,allprobs,nbirths,swappossible);
    r = randsample([1, 2, 3, 4],1,true,[p_g,p_p,p_c,p_s]);
    if r == 1 % grow
        [Tstar,birthindex] = birth(T,y,X);
        kstar = nbirthnodes(T,X); % Number of nodes that can grow
        k_d = length(terminalparents(Tstar)); % number of possible deaths in proposed model
        birthvarind = Tstar.Allnodes{birthindex}.Rule{1};
        % Number of possible variables to split on
        N_v = sum(Tstar.Allnodes{birthindex}.nSplits > 0);
        % Number of possible splits for this variable
        N_b = Tstar.Allnodes{birthindex}.nSplits(birthvarind);
        % Reversibility
        if length(Tstar.Allnodes) >= 5
            [~,swappossible] = swap(Tstar,y,X,[],0);
        else
            swappossible = [];
        end
        [nbirths,Tstar] = nbirthnodes(Tstar,X);
        [~,p_p_star,~,~] = propprobs(Tstar,allprobs,nbirths,swappossible);
        prop_ratio = log(p_p_star/k_d) - log(p_g/(kstar*N_v*N_b));
        [Tstarprior,Tstar] = prior_eval(Tstar,X);
        if ~mset
            lr = temp*(Tstar.Lliketree - T.Lliketree) + ...
                (Tstarprior - Tprior) + ...
                prop_ratio;
        end
    elseif r == 2 % prune
        [Tstar,pind] = prune(T,y,X);
        k_d = length(terminalparents(T));
        prunevarind = T.Allnodes{pind}.Rule{1};
        N_b = T.Allnodes{pind}.nSplits(prunevarind);
        N_v = sum(T.Allnodes{pind}.nSplits > 0);
        % Reversibility
        if length(Tstar.Allnodes) >= 5
            [~,swappossible] = swap(Tstar,y,X,[],0);
        else
            swappossible = [];
        end
        [nbirths,Tstar] = nbirthnodes(Tstar,X);
        kstar = nbirths;
        [p_g_star,~,~,~] = propprobs(Tstar,allprobs,nbirths,swappossible);
        prop_ratio = log(p_g_star/(kstar*N_v*N_b)) - log(p_p/k_d);
        [Tstarprior,Tstar] = prior_eval(Tstar,X);
        if ~mset
            lr = temp*(Tstar.Lliketree - T.Lliketree) + ...
                (Tstarprior - Tprior) + ...
                prop_ratio;
        end
    elseif r == 3 % change
        [Tstar,priordraw,startcont,endcont,nchange,nchange2] = change(T,y,X,p);
        [Tstarprior,Tstar] = prior_eval(Tstar,X);
        % Reversibility
        if priordraw
            if startcont
                if endcont            
                    prop_ratio = 0;
                else
                    prop_ratio = log(1/(1-p));
                end
            else % start with categorical variable
                if endcont
                    prop_ratio = log(1-p);
                else
                    prop_ratio = 0;
                end
            end
        else % not a prior draw
            if startcont
                if endcont
                    if nchange > 0 && nchange2 > 0
                        prop_ratio = log(nchange/nchange2);
                    elseif nchange == 0 && nchange2 == 0
                        prop_ratio = 0;
                    else
                        error('Should not occur')
                    end
                else
                    error('This should not happen.')
                end
            else
                error('This should never happen.')
            end
        end
        if ~mset
            lr = temp*(Tstar.Lliketree - T.Lliketree) + ...
                (Tstarprior - Tprior) + ...
                prop_ratio;
        end
    elseif r == 4; % swap
        Tstar = swap(T,y,X,[],1);
        [Tstarprior,Tstar] = prior_eval(Tstar,X);
        nT = nswaps(T,y,X);
        nTstar = nswaps(Tstar,y,X);
        prop_ratio = log(nT/nTstar);
        if ~mset
            lr = temp*(Tstar.Lliketree - T.Lliketree) + ...
                Tstarprior - Tprior + prop_ratio;
        end
    end
    if isnan(lr)
        error('Likelihood ratio is not a number')
    elseif isinf(lr) && lr > 0
        error('Infinite (positive) likelihood ratio encountered')
    end % negative infinite likeilihood ratio is ok

end
    
    