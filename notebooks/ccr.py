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
# flake8: noqa
# %load_ext nb_black
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import numpy as np

rng = np.random.default_rng(1)

# +
import os.path

try:
    input_ = snakemake.input
    output = snakemake.output
except NameError:
    input_ = """
    methods/eastbay/output/unified/ccr/YRI/dm.pkl
    methods/eastbay/output/unified/ccr/CHB/dm.pkl
    methods/eastbay/output/unified/ccr/YRI::CHB/dm.pkl
    methods/eastbay/output/simulated/1/HomSap/OutOfAfrica_3G09/YRI/n132/dm.pkl
    methods/eastbay/output/simulated/1/HomSap/OutOfAfrica_3G09/CHB/n142/dm.pkl
    methods/eastbay/output/simulated/1/HomSap/OutOfAfrica_3G09/YRI::CHB/n132/dm.pkl
""".strip().splitlines()
    input_ = [
        os.path.join("/scratch/eastbay_paper/simulation_results_full", i.strip())
        for i in input_
    ]
    output = "/dev/null"

input_d = {"real": input_[:3], "simulated": input_[3:]}

# +
# compute ccr curves for real and simulated data
import pickle
import numpy as np

T = np.geomspace(1e1, 1e5, 1000)

ccrs = {"real": [], "simulated": []}

for k in ccrs:
    dms = {}
    for pop, file in zip(["YRI", "CHB", "YRI::CHB"], input_d[k]):
        dms[pop] = pickle.load(open(file, "rb"))
    for _ in range(1000):
        dms_i = {pop: v[rng.choice(len(v))] for pop, v in dms.items()}
        ccrs[k].append(
            2
            * dms_i["YRI::CHB"].eta(T, Ne=True)
            / (dms_i["YRI"].eta(T, Ne=True) + dms_i["CHB"].eta(T, Ne=True))
        )
# -
import matplotlib.pyplot as plt
for path in input_d['real']:
    dms = pickle.load(open(path, 'rb'))
    Ne = [dm.eta(T, Ne=True) for dm in dms]
    plt.plot(T, np.median(Ne, 0))
plt.xscale('log')
plt.yscale('log')


# +
# compute ratio curves for real and simulated data
ratios = {"real": [], "simulated": []}

for k in ratios:
    dms = {}
    for pop, file in zip(["YRI", "CHB", "YRI::CHB"], input_d[k]):
        dms[pop] = pickle.load(open(file, "rb"))
    for _ in range(1000):
        dms_i = {pop: v[rng.choice(len(v))] for pop, v in dms.items()}
        Ne1, Ne2 = [dms_i[k].eta(T, Ne=True) for k in ("YRI", "CHB")]
        ratios[k].append(Ne1 / Ne2 + Ne2 / Ne1)


# +
import matplotlib as mpl

import matplotlib.pyplot as plt


def despine(ax):
    for d in ("top", "right"):
        ax.spines[d].set_visible(False)


import matplotlib.gridspec as gridspec

# Create a grid with 2 rows and 2 columns,
# but the second column is twice as wide as the first one
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1])

# Subplots
ax2 = fig.add_subplot(gs[0, 0])  # Top-left
ax1 = fig.add_subplot(gs[0, 1], sharey=ax2)
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

# fig, axs = plt.subplots(nrows=2, sharex=True)
rng = np.random.default_rng(1)
for d, ax in zip((ccrs, ratios), (ax2, ax1)):
    for k in d:
        data = d[k]
        label = k.title()
        (line,) = ax.plot(np.median(data, 0), T, label=label)
        color = line.get_color()
        qq = np.quantile(data, [0.025, 0.975], axis=0)
        ax.fill_betweenx(T, *qq, color=color, alpha=0.1)
        for q in qq:
            ax.plot(q, T, color=color, linestyle="--", linewidth=0.4)
        despine(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")


ax2.legend(bbox_to_anchor=(1.1, 0.95))
import demesdraw
import stdpopsim

G = (
    stdpopsim.get_species("HomSap")
    .get_demographic_model("OutOfAfrica_3G09")
    .model.to_demes()
)
demesdraw.tubes(G, ax=ax3, log_time=True, max_time=1e5)
ax1.set_ylim(2e1, 1e5)
ax1.set_xlabel("$N_1/N_2 + N_2/N_1$")
ax1.set_title("Ratio", y=0.96)
ax2.set_title("CCR", y=0.96)
ax1.title.set_position((0.5, 0.97))
ax2.title.set_position((0.6, 0.97))
for ax in ax1, ax2:
    ax.fill_between(ax.get_xlim(), 848, 5000, color="grey", alpha=0.15)
ax2.set_ylabel("Time (generations)")
ax2.set_xlabel("Cross-coalescence rate")
ax1.set_ylabel("")
ax3.set_ylabel("")
ax3.yaxis.set_visible(False)
ax3.spines["left"].set_visible(False)
fig.suptitle("YRI-CHB divergence time estimation")
fig.tight_layout()
fig.savefig("yri_chb_div.pdf")
# -

np.array(data).shape
