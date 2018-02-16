function [trees,LLIKES,whichout] = mallow(varargin) 
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
    trees = cell(maxtree,1);
    whichout = [];
    for ii=1:maxtree
        tmpll_old = -Inf;
        LLIKES(ii) = NaN;
        for jj=1:nout
            output = varargin{jj};
            ind = output.treesize == ii;
            llikes = output.llike(ind);
            TREES = output.Trees(ind);
            [tmpll,I] = max(llikes);
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
        end
    end
    plot(1:maxtree,LLIKES,'-o')
    xlabel('Tree Size')
    ylabel('Best p(y|T)')
end