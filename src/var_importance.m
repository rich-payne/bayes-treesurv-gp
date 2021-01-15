%    bayes-treesurv-gp provides a Bayesian tree partition model to flexibly 
%    estimate survival functions in various regions of the covariate space.
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

function V = var_importance(output, X, type)
   if strcmp(type, 'like')
       a = max(output.llike);
       log_denom = a + log(sum(exp(output.llike - a)));
       w = exp(output.llike - log_denom);
   elseif strcmp(type, 'post')
       a = max(output.llike + output.lprior);
       log_denom = a + log(sum(exp(output.llike + output.lprior - a)));
       w = exp(output.llike + output.lprior - log_denom);
   end
   w_v = zeros(size(X, 2), 1);
   for ii = 1:length(output.Trees)
     w_v = w_v + w(ii) * var_count(output.Trees{ii}, X);
   end
   [w_v_s, I] = sort(w_v, 'descend');
   V = array2table(w_v_s');
   V.Properties.VariableNames(1:size(V, 2)) = X.Properties.VariableNames(I);
end