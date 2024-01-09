import pickle

import eastbay.data
import eastbay.mcmc
import tskit
import tszip

n = int(snakemake.wildcards.num_samples)
nodes = [(2 * i, 2 * i + 1) for i in range(n)]
data = [
    eastbay.data.TreeSequenceContig(ts, nodes)
    for ts in map(tszip.decompress, snakemake.input)
]
# hold out shortest chrom for testing
test_data, *train_data = sorted(data, key=lambda c: c.L)
res = eastbay.mcmc.fit(train_data, test_data)
with open(snakemake.output[0], "wb") as f:
    pickle.dump(res, f)
