library(bnlearn)

alg <- switch(method,
              "stable" = pc.stable,
              "hiton" = si.hiton.pc,
              "maxmin" = mmpc,
              "hybrid" = hpc
)

result <- alg(x, undirected = undirected)