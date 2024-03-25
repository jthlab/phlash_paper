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

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import scienceplots
plt.style.use('science')
mpl.rcParams['font.size'] = 12
mpl.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{times}
'''

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
import phlash.plot
from phlash.util import tree_stack

# +
import glob
import os.path

## independence plot 

try:
    paths = snakemake.input.indep
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
print(df)
df.to_pickle(snakemake.output[0][:-3] + "pkl")

# +
import matplotlib.style
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained", figsize=(6.5, 2.5))
axd = fig.subplot_mosaic(
        """
        A--
        ABC
        ADE
        """,
        width_ratios=[0.33,0.33,0.33],
        height_ratios=[.05,.475,.475],
        )

axd['-'].tick_params(axis='both',which='both',bottom=False,left=False,top=False,right=False)
axd['-'].set_xticks([])
axd['-'].set_yticks([])
for d in ['right', 'top', 'bottom', 'left']:
    axd['-'].spines[d].set_visible(False)
# composite plot
ax = axd['A']
ax.boxplot(columns)
ax.set_xticks(np.arange(1, 6), labels=OVERLAPS)
ax.set_yscale('log')
ax.xaxis.set_tick_params(which='minor',bottom=False,top=False)
# ax.set_ylim(0, 1e-4)
medians = df.groupby('overlap')['rel_err'].median()
ax.plot(1 + np.arange(len(medians.index)), medians.values, linestyle="--", marker="o")
ax.set_ylabel(r"Relative error")
ax.set_xlabel("Overlap ($f$)")



import matplotlib.transforms as mtransforms
trans = mtransforms.ScaledTranslation(5/72, 15/72, fig.dpi_scale_trans)
ax.text(0.0, 0.0, "(a)", transform=ax.transAxes + trans, verticalalignment='top')


import jax.numpy as jnp
from jax import vmap

def plot_posterior(dms: list[DemographicModel], ax: "matplotlib.axes.Axes" = None, **kw):
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    dms = tree_stack(dms)
    t1, tM = jnp.quantile(dms.eta.t[:, 1:], jnp.array([0.025, 0.975]))
    t = jnp.geomspace(t1, tM, 1000)
    Ne = vmap(SizeHistory.__call__, (0, None, None))(dms.eta, t, True)
    q025, m, q975 = jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)
    ax.plot(t, m, **kw)
    ax.fill_between(t, q025, q975, alpha=0.1)



def load_file(path):
    if path.endswith("json"):
        decoder = json
        mode = "t"
    else:
        assert path.endswith("pkl"), path
        decoder = pickle
        mode = "b"
    with open(path, f"r{mode}") as f:
        return decoder.load(f)


def dump_file(obj, path):
    if path.endswith("json"):
        encoder = json
        mode = "t"
    else:
        assert path.endswith("pkl")
        encoder = pickle
        mode = "b"
    with open(path, f"w{mode}") as f:
        encoder.dump(obj, f)

## composite plot
for c, pop in zip("BCDE", ["Han", "Finnish", "Iberian", "Yoruba"]):
    ax = axd[c]
    T = np.geomspace(1e2, 2e5, 1000)
    cl_p = load_file(f"unified/{pop}/phlash/estimates.pkl")
    indep_p = load_file(f"composite/{pop}/phlash/estimates.pkl")
    plot_posterior(cl_p, ax, label="Composite")
    plot_posterior(indep_p, ax, label="Exact")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(pop, loc="center", fontsize=8, y=0.75)
    ax.set_xlim(1e2, 2e5)
    ax.sharex(axd['B'])
    ax.sharey(axd['B'])
    if c in "BC":
        ax.tick_params(labelbottom=False)
    if c in "CE":
        ax.tick_params(labelleft=False)
    if pop == "Han":
        fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="upper right", bbox_to_anchor=[0.94, 1.03])
    if c in "DE":
        ax.set_xlabel("Time")
    if c in "BD":
        ax.set_ylabel("$N_e$")
    trans = mtransforms.ScaledTranslation(-25/72, 15/72, fig.dpi_scale_trans)
    ax.text(1.0, 0.0, f"({c.lower()})", transform=ax.transAxes + trans, verticalalignment='top')


fig.savefig(snakemake.output[0], bbox_inches="tight")
# plt.subplots_adjust()
# -
