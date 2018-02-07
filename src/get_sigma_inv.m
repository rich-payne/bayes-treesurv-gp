function Sigmainv = get_sigma_inv(K,tau,l,dz)
    if isempty(dz) % implies that K == 1
        Sigmainv = 1 ./ tau^2; 
        return;
    elseif K > 2
        rho = exp(-dz ./ l);
%         B = [
%               [1; (1 + rho^2) .* ones(K-2,1); 1],...
%               [0;-rho .* ones(K-1,1)],...
%               [-rho .* ones(K-1,1);0]
%             ];
        % Sigmainv = spdiags(B,[0,1,-1],K,K);
        rhovec = [1; (1 + rho^2) .* ones(K-2,1); 1];
        rhovec2 = rho .* ones(K-1,1);
        Sigmainv = diag(rhovec) - diag(rhovec2,-1) - diag(rhovec2,1);
        Sigmainv = sparse(Sigmainv);
    elseif K == 2
        rho = exp(-dz ./ l);
        Sigmainv = [1 -rho; -rho 1];
    end    
    Sigmainv = 1./((1 - rho^2) .* tau^2) .* Sigmainv;          
end
    