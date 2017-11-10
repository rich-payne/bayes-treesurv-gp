function get_surv_tree(thetree,Y,X,ndraw,graph)
    thetree = fatten_tree(thetree,X);
    I = termnodes(thetree);
    nnodes = length(I);
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
    Ystd = Y;
    Ystd(:,1) = Ystd(:,1)/max(Ystd(:,1));
    for ii=1:length(I)
        ypart = Ystd(thetree.Allnodes{I(ii)}.Xind,:);
        [~,res] = get_marginal(ypart,thetree.K,[],thetree.eps,...
            thetree.Allnodes{I(ii)}.tau,...
            thetree.Allnodes{I(ii)}.l,...
            thetree.nugget,0);
        if ~dosubplot
            figure(ii);
        else
            subplot(s1,s2,ii);
        end
        get_surv(Ystd,res,ndraw,graph);
        title(strcat(['Node Index: ',num2str(I(ii))]));
        xlim([0,1]);
        ylim([0,1]);
    end
end