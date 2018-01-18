function pdraws = get_surv_tree(thetree,Y,X,ndraw,graph,x0,ystar,alpha)
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
    I = termnodes(thetree);
    if isempty(x0)
        theind = I;
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
    for ii=theind'
        ypart = Ystd(thetree.Allnodes{ii}.Xind,:);
        [~,res] = get_marginal(ypart,thetree.K,[],thetree.eps,...
            thetree.Allnodes{ii}.tau,...
            thetree.Allnodes{ii}.l,...
            thetree.nugget,0);
        if graph
            if ~dosubplot
                figure(cntr);
            else
                subplot(s1,s2,cntr);
            end
        end
        pdraws = get_surv(Y,res,ndraw,graph,ystar,alpha);
        if graph
            title(strcat(['Node Index: ',num2str(ii)]));
            xlim([0,Ymax]);
            ylim([0,1]);
        end
        cntr = cntr + 1;
    end
    if isempty(x0)
        pdraws = [];
    end
end