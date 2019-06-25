function V = var_importance(output, X)
   a = max(output.llike);
   log_denom = a + log(sum(exp(output.llike - a)));
   w = exp(output.llike - log_denom);
   w_v = zeros(size(X, 2), 1);
   for ii = 1:length(output.Trees)
     w_v = w_v + w(ii) * var_count(output.Trees{ii}, X);
   end
   [w_v_s, I] = sort(w_v, 'descend');
   V = array2table(w_v_s');
   V.Properties.VariableNames(1:size(V, 2)) = X.Properties.VariableNames(I);
end