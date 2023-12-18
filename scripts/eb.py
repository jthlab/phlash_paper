import pickle

import eastbay.data
import eastbay.mcmc
import tskit

n = int(snakemake.wildcards.num_samples)
nodes = [(2 * i, 2 * i + 1) for i in range(n)]
data = [
    eastbay.data.TreeSequenceDataset(ts, nodes)
    for ts in map(tskit.load, snakemake.input)
]
res = eastbay.mcmc.fit(data, options={"niter": 1000})
with open(snakemake.output[0], "wb") as f:
    pickle.dump(res, f)
