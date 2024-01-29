# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
# jax/gpu hogs all the gpu memory limiting parallelism with other parts of the pipeline.
# the results are the same between cpu and gpu.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from eastbay.hmm import psmc_ll
from eastbay.data import VcfContig, Contig
from eastbay.size_history import DemographicModel, SizeHistory
from eastbay.params import PSMCParams

# +
import glob
import os.path

try:
    paths = snakemake.input
except NameError:
    dirs = glob.glob("/scratch/eastbay_paper/pipeline/h2h/model*/rep0/simulations")
    paths = []
    for base in dirs:
        first_chrom = sorted(glob.glob("chr*.bcf", root_dir=base))[0]
        paths.append(os.path.join(base, first_chrom))
# -

paths

# +
import numpy as np
import jax

def eval_chunk_sizes(c: Contig):
    H = c.get_data(100)['het_matrix'].clip(0, 1)
    # calculate chunk size: 1/5th of chrom
    chunk_size = int(.2 * H.shape[1])
    theta = H.mean()
    dm = DemographicModel.default(pattern='16*1', theta=theta)
    alpha, ll = psmc_ll(dm, H[0])
    res = {}
    for overlap in [0, 100, 200, 500, 1000]:
        chunks = c.to_chunked(window_size=100, overlap=overlap, chunk_size=chunk_size).chunks
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
    return res


# -

res = {}
for path in paths:
    assert path.endswith("bcf")
    c = VcfContig(path, samples=["sample0"], contig=None, interval=None, _allow_empty_region=True)
    res[path] = eval_chunk_sizes(c)

# +
import pandas as pd
records = []
for path, d in res.items():
    for k in d:
        records.append({"path": path, "overlap": k, "rel_err": d[k]})

df = pd.DataFrame.from_records(records)

print(df)

# +
import matplotlib.style
matplotlib.style.use("tableau-colorblind10")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
for d in ["right", "top"]:
    ax.spines[d].set_visible(False)
ax.set_yscale('log')
# ax.set_ylim(0, 1e-4)
df.boxplot(column="rel_err", by="overlap", ax=ax)
medians = df.groupby('overlap')['rel_err'].median()
print(medians)
import numpy as np
ax.plot(1 + np.arange(len(medians.index)), medians.values, linestyle="--", marker="o")
plt.title("Exponential forgetting in PSMC")
ax.set_ylabel("Relative error of log-likelihood")
ax.set_xlabel("Length of overlap")
ax.grid(False)
fig.savefig(snakemake.output[0])
# -


