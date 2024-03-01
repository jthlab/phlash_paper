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
import numpy as np

rng = np.random.default_rng(1)

# +
import os.path

try:
    input_ = snakemake.input
    output = snakemake.output[0]
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

POPS = ["YRI", "CHB", ("YRI", "CHB")]
input_d = {k: dict(zip(POPS, getattr(input_, k)))
           for k in ("real", "simulated")}

# +
# compute ccr curves for real and simulated data
import pickle
import numpy as np

T = np.geomspace(1e1, 1e5, 1000)

# +
# the original gutenkunst demography was fitted using a factor 2x higher 
# mutation rate. therefore, to make the plots align, I need to use the 
# mutation rate assumed by that model. this would not affect real data 
# analysis so long as the simulated and real mutation rates were the same at 
# inference time.
import stdpopsim
model = stdpopsim.get_species("HomSap").get_demographic_model("OutOfAfrica_3G09")
dms = {}
for k in ["real", "simulated"]:
    d = dms[k] = {}
    for pop, file in input_d[k].items():
        d[pop] = pickle.load(open(file, "rb"))
        if k == "real":
            d[pop] = [dm.rescale(model.mutation_rate) for dm in d[pop]]

# +
ccrs = {"real": [], "simulated": []}

for k in ccrs:
    for _ in range(1000):
        dms_i = {pop: v[rng.choice(len(v))] for pop, v in dms[k].items()}
        ccrs[k].append(
            2
            * dms_i[("YRI","CHB")].eta(T, Ne=True)
            / (dms_i["YRI"].eta(T, Ne=True) + dms_i["CHB"].eta(T, Ne=True))
        )

# +
# compute ratio curves for real and simulated data
ratios = {"real": [], "simulated": []}

for k in ratios:
    for _ in range(1000):
        dms_i = {pop: v[rng.choice(len(v))] for pop, v in dms[k].items()}
        Ne1, Ne2 = [dms_i[k].eta(T, Ne=True) for k in ("YRI", "CHB")]
        ratios[k].append(Ne1 / Ne2 + Ne2 / Ne1)


# +
import matplotlib as mpl
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')

mpl.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath} 
\usepackage{amssymb}
\usepackage{times}
'''



import matplotlib.gridspec as gridspec

# Create a grid with 2 rows and 2 columns,
# but the second column is twice as wide as the first one
fig = plt.figure(figsize=(8.5-2+1, 2), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2], wspace=.25)

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
        ax.set_xscale("log")
        ax.set_yscale("log")


ax1.legend()
import demesdraw
import stdpopsim

G = (
    stdpopsim.get_species("HomSap")
    .get_demographic_model("OutOfAfrica_3G09")
    .model.to_demes()
)
cmap = dict(zip(["YRI", "CEU", "CHB"], ['#0C5DA5', '#00B945', '#FF9500']))
demesdraw.tubes(G, ax=ax3, colours=cmap, log_time=True, max_time=1e5)
ax1.set_ylim(2e1, 1e5)
ax1.set_xlabel("$N_1/N_2 + N_2/N_1$")
# ax1.set_title("Ratio")
# ax2.set_title("CCR")
# ax1.title.set_position((0.5, 0.97))
# ax2.title.set_position((0.5, 0.97))
for ax in ax1, ax2:
    ax.fill_between(ax.get_xlim(), 848, 5000, color="grey", alpha=0.15, linewidth=0)
ax2.set_ylabel("Time (generations)")
ax2.set_xlabel("Cross-coalescence rate")
ax1.set_ylabel("")
ax3.set_ylabel("")
ax3.yaxis.set_visible(False)
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
ax3.spines["left"].set_visible(False)
ax3.spines["top"].set_visible(False)
# pos = ax3.get_position() # Get current position
# new_pos = [pos.x0 - .1, pos.y0, pos.width, pos.height] # Shift left by 0.05
# ax3.set_position(new_pos) # Set new position

import matplotlib.transforms as mtransforms
for label, ax in zip('abc', [ax2, ax1, ax3]):
# label physical distance in and down:
    x = {'a': 7/72, 'b': -20/72, 'c': 15/72}[label]
    trans = mtransforms.ScaledTranslation(x, -5/72, fig.dpi_scale_trans)
    ax.text(1.0 if label == 'b' else 0., 1.0, "(" + label + ")", transform=ax.transAxes + trans,
        fontsize='medium', verticalalignment='top', fontfamily='serif',)
# fig.suptitle("YRI-CHB divergence time estimation")
# fig.tight_layout()
fig.savefig(output)
# -

np.array(data).shape
