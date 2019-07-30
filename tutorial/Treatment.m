addpath(genpath('../gpstuff'))
addpath(genpath('../src'))

% Treatment Tests
% Generate data
rng(424)
n = 1000;
x1 = rand(n,1); % prognostic
x2 = rand(n,1); % predictive (not yet implemented)
x_trt = binornd(1, .5, n, 1);
treatment = cellstr(strcat('Treatment ', num2str(x_trt + 1)));

% generate data
x_ind = x1 < .25;

% prognostic
y = wblrnd(5 * x_trt + 1, x_ind + 1);
cens = rand(n, 1) * (max(y) + 3);
y_cens = y > cens;
y(y_cens) = cens(y_cens);
Y_prognostic = [y, abs(1 - y_cens)];

% predictive (trt effect only for certain x_ind)
y = wblrnd(5 * x_trt .* x_ind + 1, 2);
cens = rand(n, 1) * (max(y) + 3);
y_cens = y > cens;
y(y_cens) = cens(y_cens);
Y_predictive = [y, abs(1 - y_cens)];

% prognostic and predictive (x_ind2 is prognostic, x_ind1 is predictive)
x_ind2 = x2 < .5;
y = wblrnd(5 .* x_trt .* x_ind + 1, 1 + x_ind2);
cens = rand(n, 1) * (max(y) + 3);
y_cens = y > cens;
y(y_cens) = cens(y_cens);
Y_prog_pred = [y, abs(1 - y_cens)];

% plot effects
% xx = 0:.1:10;
% plot(xx, 1 - wblcdf(xx, 1,  1))
% hold on;
% plot(xx, 1 - wblcdf(xx, 5,  1))
% plot(xx, 1 - wblcdf(xx, 1,  2))
% plot(xx, 1 - wblcdf(xx, 5,  2))
% hold off;

% analysis
X = table(x1, x2);
X_trt = X;
X_trt.treatment = treatment;

% Run MCMC to test trt
Tree_Surv(...
  Y_prognostic,...
  X_trt,...
  'nmcmc', 1000,...
  'burn', 0,...
  'filepath', './output_prognostic_trt/',...
  'seed', 35991,...
  'bigK', 100, ...
  'saveall', 1, ...
  'swapfreq', 2,...
  'nprint', 100, ...
  'n_parallel_temp', 4 ...
);

Tree_Surv(...
  Y_predictive,...
  X_trt,...
  'nmcmc', 10,...
  'burn', 0,...
  'filepath', './output_predictive_trt/',...
  'seed', 351,...
  'bigK', 100, ...
  'saveall', 1, ...
  'swapfreq', 2,...
  'nprint', 100, ...
  'n_parallel_temp', 4 ...
);


Tree_Surv(...
  Y_prog_pred,...
  X_trt,...
  'nmcmc', 1000,...
  'burn', 0,...
  'filepath', './output_prog_pred_trt/',...
  'seed', 25159,...
  'bigK', 100, ...
  'saveall', 1, ...
  'swapfreq', 2,...
  'nprint', 100, ...
  'n_parallel_temp', 4 ...
);

if 0 
  load('./output_prognostic_trt/mcmc_id1.mat')
  load('./output_predictive_trt/mcmc_id1.mat')
  [~, I] = max(output.llike + output.lprior);
  Treeplot(output.Trees{I})
  output.Trees{I}.Allnodes{1}
  output.Trees{I}.Allnodes{2}
  output.Trees{I}.Allnodes{3}
end