function [trees,LLIKES] = mallow(output)    
    % Mallow's C_p type plot
    maxtree = max(output.treesize);
    LLIKES = [];
    trees = cell(maxtree,1);
    for ii=1:maxtree
        ind = output.treesize == ii;
        llikes = output.llike(ind);
        TREES = output.Trees(ind);
        [tmpll,I] = max(llikes);
        if ~isempty(tmpll)
            LLIKES(ii) = tmpll;
            trees{ii} = TREES{I};
        else
            LLIKES(ii) = NaN;
        end
    end
    plot(1:maxtree,LLIKES,'-o')
    xlabel('Tree Size')
    ylabel('Best p(y|T)')
end