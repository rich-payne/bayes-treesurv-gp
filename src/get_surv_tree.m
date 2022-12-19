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
%    This function plots the posterior survival curves for each of the
%    terminal nodes of a tree.
%
%    INPUTS
%    thetree: An object of Tree class.
%    Y: A two-column matrix with the survival times in the first column
%      (on original scale) and the survival indicator in
%      the second column (0 if censored, 1 if observed).
%    X: The covariates
%    ndraw: the number of Monte Carlo draws to produce the posterior
%      survival function
%    graph: 1 to produce a graph.  0 to suppress graphing.
%    x0: If empty, a graph is produced for each terminal node.  If
%      non-empty, it must be a table with one row and the same number of
%      columns of X.  Specifying x0 produces the posterior survival function
%      for an observation with covariates x0.
%    ystar: A grid where the posterior survival function will be evaluated.
%      If empty, grid is determined automatically.
%    alpha: the significance level for the credible itnervals.
%    the_title: A string to specify the title of the graphs.  If empty, the
%      Node indexes are used to title the graphs.
%
%    OUTPUT
%    pdraws: If x0 is empty (i.e. all survival functions are plotted), then
%      all output for each terminal node from the function get_surv is
%      returned.  If x0 is non-empty, the output from the get_surv
%      function for the data in the corresponding terminal node is
%      returned.
%    term_nod_ind: the terminal node index for each element of pdraws.
function [pdraws, term_node_ind] = get_surv_tree(thetree,Y,X,ndraw,graph,x0,ystar,alpha,the_title)
    if ~exist('alpha','var')
        alpha = .05;
    end
    if ~isempty(x0)
        if size(x0,1) > 1
            error('x0 must have only one row');
        elseif size(x0,2) ~= size(X,2)
            error('x0 must have the same number of columns as X');
        end
    end
    thetree = fatten_tree(thetree,X);
    term_node_ind = termnodes(thetree);
    if isempty(x0)
        theind = term_node_ind;
    else
        [~,theind] = get_termnode(thetree,x0);
    end    
    nnodes = length(theind);
    if nnodes <= 9
        dosubplot = 1;
    else 
        dosubplot = 0;
    end
    if nnodes == 1
        s1 = 1;
        s2 = 1;
    elseif nnodes == 2
        s1 = 1;
        s2 = 2;
    elseif nnodes <= 4
        s1 = 2;
        s2 = 2;
    elseif nnodes <= 6
        s1 = 2;
        s2 = 3;
    elseif nnodes <= 9
        s1 = 3;
        s2 = 3;
    end
    Ymax = max(Y(:,1));
    Ystd = Y;
    Ystd(:,1) = Ystd(:,1)/Ymax;
    
    cntr = 1;
    pdraws = cell(nnodes, 1);
    for ii=theind'
        ypart = Ystd(thetree.Allnodes{ii}.Xind,:);
        [~,res] = get_marginal(ypart,thetree.K,[],thetree.eps,...
            thetree.Allnodes{ii}.tau,...
            thetree.Allnodes{ii}.l,...
            thetree.nugget,0);
        if graph
            if ~dosubplot
                figure(cntr);
            elseif s1 == 1 && s2 == 1 % Do no subplot command
            else
                subplot(s1,s2,cntr);
            end
        end
        pdraws{cntr} = get_surv(Y,res,ndraw,graph,ystar,alpha);
        if graph
            if isempty(the_title)
                title(strcat(['Node Index: ',num2str(ii)]));
            else
                title(the_title)
            end
            xlim([0,Ymax]);
            ylim([-.05,1.05]);
        end
        cntr = cntr + 1;
    end
end