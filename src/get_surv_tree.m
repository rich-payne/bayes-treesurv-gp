function get_surv_tree(thetree,Y,X,ndraw,graph)
    thetree = fatten_tree(thetree,X);
    I = termnodes(thetree);
    Ystd = Y;
    Ystd(:,1) = Ystd(:,1)/max(Ystd(:,1));
    for ii=1:length(I)
        ypart = Ystd(thetree.Allnodes{I(ii)}.Xind,:);
        [~,res] = get_marginal(ypart,thetree.K,[],thetree.eps,...
            thetree.Allnodes{I(ii)}.tau,...
            thetree.Allnodes{I(ii)}.l,...
            thetree.nugget,0);
        figure(ii);
        get_surv(Ystd,res,ndraw,graph);
        title(strcat(['Node Index: ',num2str(I(ii))]));
    end
end