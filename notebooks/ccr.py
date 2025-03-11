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

POPS = ["YRI", "CHB", ("YRI", "CHB"), "merged"]
input_d = {k: dict(zip(POPS, getattr(input_, k)))
           for k in ("real", "simulated")[:1]}

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
for k in ["real", "simulated"][:1]:
    d = dms[k] = {}
    for pop, file in input_d[k].items():
        d[pop] = pickle.load(open(file, "rb"))
        if k == "real":
            d[pop] = [dm.rescale(model.mutation_rate) for dm in d[pop]]

# +
ccrs = {"real": []}

for k in ccrs:
    for _ in range(1000):
        dms_i = {pop: v[rng.choice(len(v))] for pop, v in dms[k].items()}
        ccrs[k].append(
            2
            * dms_i[("YRI","CHB")].eta(T)
            / (dms_i["YRI"].eta(T) + dms_i["CHB"].eta(T))
        )

# +
# compute ratio curves for real and simulated data
ratios = {"real": []}

for k in ratios:
    for _ in range(1000):
        dms_i = {pop: v[rng.choice(len(v))] for pop, v in dms[k].items()}
        Ne1, Ne2, Ne_comb = [dms_i[k].eta(T) for k in ("YRI", "CHB", "merged")]
        x = Ne_comb / (.5 * (Ne1 + Ne2))
        ratios[k].append(x)


# +
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import scienceplots
plt.style.use('science')
mpl.rcParams['font.size'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{times}
'''



import matplotlib.gridspec as gridspec

# Create a grid with 2 rows and 2 columns,
# but the second column is twice as wide as the first one
# fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(6.5, 2.5), layout="constrained")
# (ax2, ax1, ax3) = axs
fig = plt.figure(figsize=(4.5, 2.5), layout="constrained")
axd = fig.subplot_mosaic("BC", sharey=True, width_ratios=[2, 1])
# ax2 = axd["A"]
ax1 = axd["B"]
ax3 = axd["C"]
# Subplots
# fig, axs = plt.subplots(nrows=2, sharex=True)
rng = np.random.default_rng(1)
for d, ax in list(zip((ccrs, ratios), (None, ax1)))[1:]:
    for k in ["real"]:
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


# ax1.legend()
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
ax1.set_xlabel(r"$\frac{2\eta_{\text{Combined}}}{\eta_{\text{YRI}} + \eta_{\text{CHB}}}$")
# ax1.set_title("Ratio")
# ax2.set_title("CCR")
# ax1.title.set_position((0.5, 0.97))
# ax2.title.set_position((0.5, 0.97))
for ax in (ax1, None)[:1]:
    ax.fill_between(ax.get_xlim(), 848, 5000, color="grey", alpha=0.15, linewidth=0)
#$ ax2.set_ylabel("Time (generations)")
#$ ax2.set_xlabel(r"$\frac{2\eta_{\text{Between}}}{\eta_{\text{YRI}} + \eta_{\text{CHB}}}$ (CCR)")
# ax1.set_ylabel("")
ax3.set_ylabel("")
ax3.yaxis.set_visible(False)
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
for d in ["top", "right", "left", "bottom"]:
    ax3.spines[d].set_visible(False)
# pos = ax3.get_position() # Get current position
# new_pos = [pos.x0 - .1, pos.y0, pos.width, pos.height] # Shift left by 0.05
# ax3.set_position(new_pos) # Set new position

import matplotlib.transforms as mtransforms
# for label, ax in zip('abc', [ax2, ax1, ax3]):
for label, ax in zip('ab', [ax1, ax3]):
# label physical distance in and down:
    x = {'a': 7/72, 'b': -20/72, 'c': 13/72}[label]
    trans = mtransforms.ScaledTranslation(x, -5/72, fig.dpi_scale_trans)
    ax.text(1.0 if label == 'b' else 0., 1.0, "(" + label + ")", transform=ax.transAxes + trans,
        verticalalignment='top')
# fig.suptitle("YRI-CHB divergence time estimation")
# fig.tight_layout()
fig.savefig(output, bbox_inches="tight")
# -

np.array(data).shape
