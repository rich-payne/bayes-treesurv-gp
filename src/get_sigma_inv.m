function Sigmainv = get_sigma_inv(K,tau,l,dz)
    rho = exp(-dz ./ l);
    B = [
          [1; (1 + rho^2) * ones(K-2,1); 1],...
          [0;-rho * ones(K-1,1)],...
          [-rho * ones(K-1,1);0]
        ];
    Sigmainv = spdiags(B,[0,1,-1],K,K);
    Sigmainv = 1./((1 - rho^2) .* tau^2) .* Sigmainv;
end
    