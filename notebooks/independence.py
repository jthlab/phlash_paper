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
import numpy as np
import os
# jax/gpu hogs all the gpu memory limiting parallelism with other parts of the pipeline.
# the results are the same between cpu and gpu.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from phlash.hmm import psmc_ll
from phlash.data import VcfContig, Contig
from phlash.size_history import DemographicModel, SizeHistory
from phlash.params import PSMCParams

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

import pandas as pd
records = []
for path in paths:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    d['path'] = path
    records.append(d)

df = pd.DataFrame.from_records(records)
OVERLAPS = [0, 100, 200, 500, 1000]
columns = [df['rel_err'][df['overlap'] == oo].to_numpy() for oo in OVERLAPS]

# +
import matplotlib.style
import matplotlib.pyplot as plt
plt.set_cmap("Set1")

s = (8.5 - 2) / 2
fig, ax = plt.subplots(figsize=(s, s), dpi=300, layout="constrained")
ax.boxplot(columns)
ax.set_xticks(np.arange(1, 6), labels=OVERLAPS)
ax.set_yscale('log')
# ax.set_ylim(0, 1e-4)
medians = df.groupby('overlap')['rel_err'].median()
ax.plot(1 + np.arange(len(medians.index)), medians.values, linestyle="--", marker="o")
fig.suptitle("Exponential forgetting in PSMC")
ax.set_ylabel("Relative error of log-likelihood")
ax.set_xlabel("Length of overlap")
ax.grid(False)
fig.savefig(snakemake.output[0], bbox_inches="tight")
# -
