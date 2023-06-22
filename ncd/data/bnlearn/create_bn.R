library(bnlearn)

set.seed(seed)
load(file_path)
adj <- as.data.frame(amat(bn))
X <- rbn(bn, n=n_samples)



