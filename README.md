# bayes-treesurv-gp
This code performs Bayesian conditional survival analysis by using a tree
to partition the covariate space and then fits flexible survival
curves in each partition element. Details are outlined in the paper entitled "A Bayesian Survival Treed Hazards Model Using Latent Gaussian Processes" which is accepted in *Biometrics*.

NOTE: Be sure the GPstuff toolbox is included in the Matlab path.
GPstuff must be downloaded separately from:
http://research.cs.aalto.fi/pml/software/gpstuff/

Jarno Vanhatalo, Jaakko Riihimäki, Jouni Hartikainen, Pasi Jylänki, Ville Tolvanen, and Aki Vehtari (2013). GPstuff: Bayesian Modeling with Gaussian Processes. Journal of Machine Learning Research, 14(Apr):1175-1179. (Available at http://jmlr.csail.mit.edu/papers/v14/vanhatalo13a.html)
