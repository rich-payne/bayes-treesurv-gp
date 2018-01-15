function [truemarginal,marges,thegrid] = get_truemarginal(Y,res,dtau,dl,dmu,ntau,nl,nmu)
    gridmu = linspace(res.mu - dmu .* nmu./2,res.mu + dmu .* nmu ./2,nmu);
    gridtau = linspace(res.tau - dtau .* ntau/2,res.tau + dtau .* ntau/2,ntau);
    gridl = linspace(res.l - dl .* nl/2,res.l + dl .* res.l/2,nl);

    % Restrict to only positive values of tau and l
    gridtau(gridtau <= 0) = [];
    gridl(gridl <= 0) = [];

    thegrid = combvec(gridmu,gridtau,gridl);

    marges = zeros(size(thegrid,2),1);
    for ii=1:size(thegrid,2)
        tmpmarg = get_marginal(Y,20,[],1e-10,thegrid(2,ii),thegrid(3,ii),thegrid(1,ii),1e-10,0);
        marges(ii) = tmpmarg;
    end
    truemarginal = dmu .* dtau .* dl .* sum(exp(marges));
end