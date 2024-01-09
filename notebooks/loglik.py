# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
# jax/gpu hogs all the gpu memory limiting parallelism. 
# the results are the same between cpu and gpu.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from importlib.metadata import version
print(version('eastbay'))

import stdpopsim
import eastbay.data
import numpy as np
import jax
from eastbay.params import PSMCParams

# +
seed = 92682

contig = "2L" if snakemake.wildcards.species in ["DroMel", "AnoGam"] else "1"

truth, data = eastbay.data.stdpopsim_dataset(
    snakemake.wildcards.species,
    snakemake.wildcards.demographic_model,
    snakemake.wildcards.population,
    included_contigs=[contig],
    n_samples=1, options=dict(seed=seed, length_multiplier=1.0)
)
# -

# exact likelihood (no chunking)
from eastbay.hmm import psmc_ll
from eastbay.size_history import DemographicModel, SizeHistory

H = data[0].get_data(100)['het_matrix'].clip(0, 1)
# calculate chunk size: 1/5th of chrom
chunk_size = int(.2 * H.shape[1])
res = {}

theta = H.mean()
dm = DemographicModel.default(pattern='16*1', theta=theta)
alpha, ll = psmc_ll(dm, H[0])

for overlap in [0, 100, 200, 500, 1000]:
    chunks = data[0].chunk(window_size=100, overlap=overlap, chunk_size=chunk_size)['chunks']
    warmup_chunks, data_chunks = np.split(chunks[1:], [overlap], axis=1)

    pp = PSMCParams.from_dm(dm)
    # the first chunk overlaps into the second chunk, so we don't want to double count
    a0, ll0 = psmc_ll(dm, chunks[0])
    # the handle the rest of the chunks
    pis = jax.vmap(lambda pp, d: psmc_ll(pp, d)[0], (None, 0))(pp, warmup_chunks)  # (I, M)
    pps = jax.vmap(lambda pi: pp._replace(pi=pi))(pis)
    _, ll1 = jax.vmap(psmc_ll, (0, 0))(pps, data_chunks)

    ll_par = ll0+ll1.sum()
    re = abs((ll_par - ll) / ll)
    res[overlap] = float(re)

import pickle
with open(snakemake.output[0], "wb") as f:
    pickle.dump(res, f)
