%     This file is part of bayes-treed-cde.
% 
%     bayes-treed-cde is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     bayes-treed-cde is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with bayes-treed-cde.  If not, see <http://www.gnu.org/licenses/>.
%
%     Copyright 2016-2017, Richard Payne

classdef Nodes
    properties
        Depth % Depth of node
        Id % Id for the node
        Parent % Id of the parent node
        Lchild % Id of left child
        Rchild % Id of right child
        Rule % Split rule for the node, cell of length 2 which
             %    contains the variable number/column and rule, respectively.
        Xind % Index of which data values pass through node
        Llike % Log-Likelihood contribution of data in this node
        Splitvals % The possible rules available for splitting on each variable
        nSplits % The number of rules available for splitting on each variable
        Updatellike % 0 if no llike update/computation needed, 
                       % 1 if it is needed
        Updatesplits % 0 if no update of the splits is necessary, 1 if needed
        tau % EB scale hyperparameters 
        l % EB length hyperparamter
    end
    methods
        % Constructor
        function obj = Nodes(Id,Parent,Lchild,Rchild,Rule,Xind,Depth) % Add in an Xind later?
            % Id: ID of the new node (integer)
            % Parent: ID of parent (integer)
            % Lchild: ID of left child (integer)
            % Rchild: ID of right child (integer)
            % Rule: A cell containing column number and the rule
            % Xind: An index of which data points descend to the node
            % Depth: How many levels deep the rule is (root node is 0)
            
            % Assign ID
            if isa(Id,'numeric')
                obj.Id = Id;
            else
                error('Id must be numeric')
            end
            % Assign Parent
            if isa(Parent,'numeric')
                obj.Parent = Parent;
            else
                error('Parent Id must be numeric')
            end
            % Assign Children
            if isa(Lchild,'numeric')
                obj.Lchild = Lchild;
            else
                error('Lchild must be numeric')
            end
            if isa(Rchild,'numeric')
                obj.Rchild = Rchild;
            else
                error('Rchild must be numeric')
            end
            % Assign the split rule for the node
            if iscell(Rule)
                if length(Rule) == 2
                    obj.Rule = Rule;
                else 
                    error('Rule must be of length 2.')
                end
            elseif isempty(Rule)
            else 
                error('Rule must be empty or a cell object of length 3.')
            end
            % Assign the index for which data values pass through this
            %   node.
            if isa(Xind,'numeric')
                obj.Xind = Xind;
            elseif isempty(Xind)
            else
                error('Xind must be numeric or empty')
            end
            % Assign Depth
            if isa(Depth,'numeric')
                obj.Depth = Depth;
            else
                error('Depth must be numeric')
            end
            
            obj.tau = [];
            obj.l = [];
            % Llike and splits needs to be computed
            obj.Updatellike = 1;
            obj.Updatesplits = 1;
        end
        
        % Draw a new rule from the prior
        % Node is an object of class 'Nodes' for which the rule is to be
        %   drawn
        function out = drawrule(node) % obj is a Tree class object
            out = node;
            if out.Updatesplits == 1 % Update splits if necessary
                error('Splits must be updated prior to calling drawrule.')
            end            
            if sum(out.nSplits) > 0 % See if there are any rules available
                % Randomly sample a variable from variables with
                %   available split rules
                vindex = find(out.nSplits > 0);
                vind = vindex(randsample(length(vindex),1)); % Variable index
                % Now randomly sample a rule from the selected variable
                svals = out.Splitvals{vind};
                if isa(svals,'cell')
                    newrule = svals{randsample(length(svals),1)};
                else
                    newrule = svals(randsample(length(svals),1));
                end
                out.Rule = {vind,newrule};
            end % else leave rule empty
        end
        
        
        
        % Calculate the log-likelihood of the node
        % obj: Object of class 'Nodes' for which the likelihood will be
        %      calculated
        % thetree: The tree to which the node 'obj' belongs
        % y: the dependent variable
        function out = loglikefunc(obj,thetree,y)
            out = obj;
            ypart = y(obj.Xind,:);
            % use parent values of a and theta as starting points
            pid = obj.Parent;
            if ~isempty(pid)
                pind = nodeind(thetree,pid);
                pnode = thetree.Allnodes{pind};
                tau_start = pnode.tau;
                l_start = pnode.l;
            else
                tau_start = 10;
                l_start = .01;
            end
            try
                [marg_y,res] = get_marginal(ypart,thetree.K,[],thetree.eps,...
                    tau_start,l_start,thetree.nugget,thetree.EB);
            catch
                if (tau_start ~= 10) || (l_start ~= .01) 
                    tau_start = 10;
                    l_start = .01;
                    [marg_y,res] = get_marginal(ypart,thetree.K,[],thetree.eps,...
                        tau_start,l_start,thetree.nugget,thetree.EB);
                end
            end
            out.tau = res.tau;
            out.l = res.l;
            out.Llike = marg_y;
        end
        
        % Obtain possible splitting values for a node
        % obj: object of class 'Nodes'
        % X: The matrix of covariates
        % leafmin: the minimum number of observations required at a
        %          terminal node.
        function out = getsplits(obj,X,leafmin)
            out = obj;
            p = size(X,2);
            Xsub = X(obj.Xind,:);
            nsub = length(obj.Xind);
            % Determine which variables are available for a split
            %   and how many split points are available.
            nSplit = zeros(p,1);
            splitvals = cell(p,1);
            for ii=1:p
                xsub = Xsub{:,ii};
                nsplits = 0;
                svals = [];
                if isa(xsub,'numeric')
                    if length(unique(xsub)) > 1
                        thetab = tabulate(xsub);
                        % Remove any zero counts (can happen with only
                        %     positive integer x-variables
                        thetab = thetab(thetab(:,2) > 0,:);
                        % Use the cumulative sums to figure out what split
                        %   variables are possible
                        csum = cumsum(thetab(:,2));
                        %rcsum = cumsum(thetab(:,2),'reverse');
                        % Index on the table of which values you can split
                        % on.
                        splitind = (csum >= leafmin) & ((nsub - csum) >= leafmin);
                        nsplits = sum(splitind);
                        svals = thetab(splitind,1);
                        if min(size(svals)) == 0
                            svals = [];
                        end
                    end
                elseif isa(xsub,'cell')
                    thetab = tabulate(xsub);
                    if length(unique(xsub)) > 1
                        % thetab = tabulate(xsub);
                        % group = thetab(:,1);
                        ngroup = size(thetab,1);
                        % More comptuationally demanding case
                        for jj = 1:ngroup
                            cmat = combnk(1:ngroup,jj);
                            for kk = 1:size(cmat,1)
                                ngroup1 = sum(cell2mat(thetab(cmat(kk,:),2)));
                                ngroup2 = nsub - ngroup1;
                                if ngroup1 >= leafmin && ngroup2 >= leafmin
                                    nsplits = nsplits + 1;
                                    svals{nsplits} = thetab(cmat(kk,:),1);
                                end
                            end
                        end
                        nsplits = nsplits/2;
                        if mod(nsplits,1) ~= 0
                            nsplits
                            error('nsplits should be an integer')
                        end
                    elseif length(unique(xsub)) == 1 
                    else
                        c1 = length(unique(xsub))
                        c2 = all(cell2mat(thetab(:,2)) > leafmin)
                        thetab
                        % orig = obj
                        % newone = out
                        error('Unexpected case.')
                    end
                else
                    error('Data type received was not expected.')
                end
                
                % Check for consistency
                if nsplits > 0 && isempty(svals)
                    error('Empty split values but positive number of splits.')
                end
                nSplit(ii) = nsplits;
                splitvals{ii} = svals;
            end
            out.nSplits = nSplit;
            out.Splitvals = splitvals;
            out.Updatesplits = 0;
        end
    end
end