function out = get_surv(Y,res,ndraw,graph)
    nstar = 100;
    ystar = linspace(.001,max(Y(:,1)),nstar); % Grid to evaluate the survival function
    binind = zeros(nstar,1);
    K = length(res.s) - 1;
    for ii=1:nstar
        [~,I] = max( ystar(ii) < res.s(2:(K+1)) );
        binind(ii) = I;
    end
    thesurv = zeros(nstar,1);
    samps = chol(res.Omegainv) \ normrnd(0,1,length(res.f),ndraw);
    samps = (samps + res.f)';
    SURV = zeros(size(samps,1),nstar);
    for ii=1:size(samps,1)
        f = samps(ii,:)';
        for jj=1:nstar
            a = ystar(jj) - res.s(binind(jj));
            if binind(jj) > 1
                thediff = diff(res.s');
                b = sum(thediff(1:(binind(jj)-1)) .* exp(f(1:(binind(jj)-1))));
            else
                b = 0;
            end
            thesurv(jj) = exp( - exp(f(binind(jj)))*a - b );
        end
        SURV(ii,:) = thesurv;
    end
    out.surv = SURV;
    out.ystar = ystar;
    
    pmean = mean(SURV);
    qtiles = quantile(SURV,[.025,.975],1);

    out.pmean = pmean;
    out.CI = qtiles;
    if graph
        plot(ystar,pmean,'-k',ystar,qtiles(1,:),'--k',ystar,qtiles(2,:),'--k')
    end
    
    
    
end