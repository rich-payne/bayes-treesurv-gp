%    bayes-treesurv-gp provides a Bayesian tree partition model to flexibly 
%    estimate survival functions in various regions of the covariate space.
%    Copyright (C) 2017-2018  Richard D. Payne
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    This function returns the precision matrix for K equally spaced 
%    process values with magnitude tau, and length-scale l, with distance
%    dz between each location.
%
%    INPUTS
%    K: the number of bins modeling the hazard function
%    tau: the magnitude hyperparameter
%    l: the length-scale hyperparameter
%    dz: the distance between the bin centers
%
%    OUTPUT
%    Sigmainv: the precision matrix

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
    