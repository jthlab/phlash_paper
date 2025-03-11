# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
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
import numpy as np
import jax
jax.config.update('jax_platforms', 'cpu')

# %load_ext nb_black
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
mpl.rcParams['mathtext.fontset'] = 'stix'



import pickle
from pathlib import Path
import glob
try:
    base = ""
    infiles = snakemake.input.estimates
except NameError:
    base = "/mnt/turbo/jonth/pipeline/"
    infiles = list(glob.glob("adna/*/*filtered/phlash/estimates.pkl", root_dir=base))

infiles

fig, axs = plt.subplots(figsize=(6.5, 4), nrows=2, ncols=2 ,sharex=True, sharey=True, layout="constrained")

T = np.geomspace(1e1, 1e6)
def plot_posterior(ax, dms, post, **kw):
    Nes = [d.eta(T, Ne=True) for d in dms]
    q0, Ne, q1 = np.quantile(Nes, [0.025, 0.5, 0.975], 0)
    ax.plot(T, Ne, **kw)
    if post:
        ax.fill_between(T, q0, q1, alpha=0.3)
    
for i, (ax, pop) in enumerate(zip(axs.reshape(-1), ("Altai", "Chagyrskaya", "Denisovan", "Vindija"))):
    for f in "", "un":
        lbl = f"{f}filtered"
        dms = pickle.load(open(base + f"adna/{pop}/{f}filtered/phlash/estimates.pkl", "rb"))
        plot_posterior(ax, dms, f == "", label=lbl.title())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(pop)
    if i == 0:
        ax.legend()

fig.supxlabel("Time", fontsize=12)
fig.supylabel("$N_e(t)$", fontsize=12)

try:
    fig.savefig(snakemake.output.combined)
except NameError:
    pass

# +
try:
    mutation_counts_f = snakemake.input.mutation_counts
except NameError:
    mutation_counts_f = "/mnt/turbo/jonth/pipeline/adna/deamination.pkl"

mutation_counts = pickle.load(open(mutation_counts_f, "rb"))
    
import pandas as pd
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt

pd.options.plotting.backend = "matplotlib"

# Desired canvas size
canvas_width, canvas_height = 6.5, 2  # in inches

# Estimate extra space for margins - you may need to adjust these
left_margin = 0.1  # Fraction of figure width
right_margin = 0.05  # Fraction of figure width
bottom_margin = 0.1  # Fraction of figure height
top_margin = 0.1  # Fraction of figure height

# Calculate figure size
figure_width = canvas_width / (1 - left_margin - right_margin)
figure_height = canvas_height / (1 - bottom_margin - top_margin)

# Create figure
fig, (ax1, ax2) = plt.subplots(figsize=(6.5, 2.5), ncols=2, layout="constrained")

# Adjust subplot parameters to make the drawing canvas fit the desired size

df = pd.DataFrame.from_records(
    {"ancestral": a, "derived": b, "p": p} for (a, b), p in mutation_counts.items()
)
df = df[df["ancestral"] != df["derived"]]
anc = df["ancestral"]
der = df["derived"]
df["Relative Frequency"] = df["p"] / df["p"].sum()
# df['a'] = np.minimum(anc, der)
# df['b'] = np.maximum(anc, der)
# df['Mutation Type'] = df['a'] + "↔" + df["b"]
df["Mutation Type"] = df["ancestral"] + r"$\to$" + df["derived"]
df.set_index("Mutation Type")
df["color"] = np.select(
    [
        (anc == "C") & (der == "T"),
        (anc == "T") & (der == "C"),
        (anc == "A") & (der == "G"),
        (anc == "G") & (der == "A"),
    ],
    ["tab:blue"] * 4,
    "lightgray",
)
df = df.sort_values("Relative Frequency", ascending=False)
df.plot.bar(
    "Mutation Type",
    "Relative Frequency",
    color=df["color"],
    # title="Mutation types in ancient samples",
    ax=ax1,
    legend=None,
)
ax1.set_ylabel("Proportion")
ax1.set_xlabel("")

pop = "Vindija"
for f in "", "un":
    lbl = f"{f}filtered"
    dms = pickle.load(open(base + f"adna/{pop}/{f}filtered/phlash/estimates.pkl", "rb"))
    plot_posterior(ax2, dms, f == "", label=lbl.title())
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1e1, 1e6)
# ax2.set_title(pop)
ax2.legend(loc="lower right")
ax2.set_ylabel('$N_e(t)$')
ax2.set_xlabel("Time")

        
import matplotlib.transforms as mtransforms
trans = mtransforms.ScaledTranslation(-20/72, -15/72, fig.dpi_scale_trans)
ax1.text(1.0, 1.0, "(a)", transform=ax1.transAxes + trans)
trans = mtransforms.ScaledTranslation(10/72, -15/72, fig.dpi_scale_trans)
ax2.text(0.0, 1.0, "(b)", transform=ax2.transAxes + trans)
# ax.yaxis.set_tick_params(which='minor', size=5, width=.5)
try:
    fig.savefig(snakemake.output.main, bbox_inches="tight")
except NameError:
    pass
# plt.savefig("deamination.pdf")

