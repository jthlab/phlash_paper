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

# %load_ext nb_black

# +
import matplotlib as mpl
import numpy as np
mpl.style.use("seaborn-v0_8-white")
import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)


# +
import pickle
import tszip

chr1p = tszip.decompress(
    "/scratch/unified/hgdp_tgp_sgdp_high_cov_ancients_chr1_p.dated.trees.tsz"
)
# metadata = {
#     'individuals': list(chr1p.individuals()),
#     'populations': list(chr1p.populations()),
#     'individual_populations': chr1p.individual_populations
# }

# for p in metadata['populations']:
#     p.metadata = json.loads(p.metadata)

# pickle.dump(metadata, open('/scratch/unified/metadata.pkl', 'wb'))

metadata = pickle.load(open('/scratch/unified/metadata.pkl', 'rb'))

ancient_pops = [
    p
    for p in metadata['populations']
    if p.metadata.get("super_population") in ("Max Planck", "Afanasievo")
]
ancient_pops
# -

(inds,) = (
    (metadata['individual_populations'][:, None] == [p.id for p in ancient_pops])
    .any(1)
    .nonzero()
)
ancient_samples = np.array([metadata['individuals'][i].nodes for i in inds])
nodes = [n for i in inds for n in metadata['individuals'][i].nodes]

# +
import tskit
from collections import Counter

mutation_counts = Counter()
A = np.array(tskit.ALLELES_ACGT)
I = np.arange(2 * chr1p.num_individuals)
not_ancient_mask = np.all(I[:, None, None] != ancient_samples[None], (1, 2))
for v in chr1p.variants(
    samples=ancient_samples.reshape(-1), alleles=tskit.ALLELES_ACGT
):
    a = v.site.ancestral_state
    adna_alleles = A[v.genotypes].reshape(-1)
    mutation_counts.update([(a, b) for b in adna_alleles])

# +
import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.style.use("seaborn-v0_8-white")
import matplotlib.pyplot as plt

pd.options.plotting.backend = "matplotlib"
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
df["Mutation Type"] = df["ancestral"] + r"→" + df["derived"]
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
ax = df.plot.bar(
    "Mutation Type",
    "Relative Frequency",
    color=df["color"],
    title="Mutation types in ancient samples",
    legend=None,
)
ax.set_ylabel("Proportion")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("")

ax.set_yticks(np.linspace(0, 0.2, 5))
ax.yaxis.set_tick_params(which="major", size=5, width=1)
# ax.yaxis.set_tick_params(which='minor', size=5, width=.5)
plt.tight_layout()
plt.savefig("deamination.pdf")
# -

# for each A, C, G, T in the ancient samples, what is the conditional distribution of A, C, G, T in the
# modern samples
D = np.zeros([4, 4], dtype=int)
A = np.array(tskit.ALLELES_ACGT)
I = np.arange(2 * chr1p.num_individuals)
not_ancient_mask = np.all(I[:, None, None] != ancient_samples[None], (1, 2))
for v in chr1p.variants(alleles=tskit.ALLELES_ACGT):
    adna_alleles = v.genotypes[~not_ancient_mask]
    modern_alleles = v.genotypes[not_ancient_mask]
    D += (
        np.bincount(adna_alleles, minlength=4)[:, None]
        * np.bincount(modern_alleles, minlength=4)[None, :]
    )

# +
import seaborn as sns

Dn = D.copy().astype(float)
np.fill_diagonal(Dn, 0.0)
Dn /= Dn.sum(1, keepdims=True)
np.fill_diagonal(Dn, np.nan)
ax = sns.heatmap(
    Dn,
    annot=True,
    fmt=".2%",
    cmap="viridis",
    xticklabels=["A", "C", "G", "T"],
    yticklabels=["A", "C", "G", "T"],
)

plt.ylabel("Ancient samples")
plt.xlabel("Modern samples")
plt.title("Relative Frequency Heatmap")
plt.show()
# -

deam_modern_maf = {}
A = np.array(tskit.ALLELES_ACGT)
I = np.arange(2 * chr1p.num_individuals)
modern_mask = np.all(I[:, None, None] != ancient_samples[None], (1, 2))
for v in chr1p.variants(alleles=tskit.ALLELES_ACGT):
    if len(np.unique(v.genotypes)) > 2:
        # restrict to biallelic for simplicity
        continue
    a = v.site.ancestral_state
    adna_gt = A[v.genotypes[~modern_mask]]
    if np.all(adna_gt == a):
        continue
    b = adna_gt[adna_gt != a].reshape(-1)[0]
    key = frozenset({a, b})
    deam_modern_maf.setdefault(key, [])
    modern_maf = (A[v.genotypes[modern_mask]] != a).sum()
    deam_modern_maf[key].append(modern_maf)


# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert dictionary to long format DataFrame
data_long = pd.DataFrame(
    [
        ("↔".join(sorted(key)), val)
        for key, values in deam_modern_maf.items()
        for val in values
    ],
    columns=["Key", "Value"],
)

v = data_long["Value"]
data_long.loc[v > v.max() / 2, "Value"] = v.max() - v[v > v.max() / 2]

order = ["C↔T", "A↔G"]
order.extend(sorted([x for x in data_long["Key"].unique() if x not in order]))
colors = ["tab:blue"] * 2 + ["lightgrey"] * 4

for a in order, colors:
    a[1], a[3] = a[3], a[1]

g = sns.FacetGrid(
    data_long,
    col="Key",
    hue="Key",
    col_wrap=3,
    sharex=True,
    sharey=True,
    col_order=order,
    palette=dict(zip(order, colors)),
)
g.set_titles("{col_name}")
g.set_ylabels("Density")
g.set(xlim=(0, 20))
ax = g.map(plt.hist, "Value", density=True, bins=np.r_[np.arange(0, 100), 1000, 5000])
g.set_xlabels("")
g.fig.subplots_adjust(bottom=0.2)
g.fig.text(0.5, -0.02, "MAF in modern samples", ha="center", fontsize=12)
g.fig.text(
    0.66,
    -0.0,
    "(Folded frequency spectrum; truncated at MAF=20)",
    fontsize=8,
    color="#555",
)
g.fig.suptitle("Ancient C↔T enrichment at low modern MAF")
g.fig.tight_layout()
g.fig.savefig("enrichment.pdf")
# -

ancient_samples

# +
import tszip
import numpy as np
import tskit

ancient_samples = np.array(
    [
        [7508, 7509],
        [7510, 7511],
        [7512, 7513],
        [7514, 7515],
        [7516, 7517],
        [7518, 7519],
        [7520, 7521],
        [7522, 7523],
    ],
    dtype=int,
)


def filtered_het_matrix(chrom_path):
    """create a het matrix for each sample pair in ancient samples, filtering out
    any a<->g or c<->t that are above an MAF cutoff in modern samples"""
    chrom = tszip.decompress(chrom_path)
    ancient_nodes = list(map(list, ancient_samples))
    L = chrom.sequence_length
    w = 100
    ret = np.zeros([len(ancient_samples), int(L // w + 1)], dtype=np.int8)
    A = np.array(tskit.ALLELES_ACGT)
    I = np.arange(2 * chrom.num_individuals)
    modern_mask = np.all(I[:, None, None] != ancient_samples[None], (1, 2))
    DEAM = {frozenset({"C", "T"}), frozenset({"A", "G"})}
    for v in chrom.variants(alleles=tskit.ALLELES_ACGT):
        if len(np.unique(v.genotypes)) > 2:
            # restrict to biallelic for simplicity
            continue
        a = v.site.ancestral_state
        adna_gt = A[v.genotypes[~modern_mask]]
        if np.all(adna_gt == a):
            continue
        b = adna_gt[adna_gt != a].reshape(-1)[0]
        key = frozenset({a, b})
        deam = key in DEAM
        modern_maf = (A[v.genotypes[modern_mask]] != a).sum()
        if DEAM and modern_maf <= 5:
            continue
        ell = int(v.position / w)
        g = v.genotypes[ancient_samples]
        ret[:, ell] += g[:, 0] != g[:, 1]
    return (chrom_path, ret)


# +
from concurrent.futures import ProcessPoolExecutor
import glob
import pickle
import gzip

# takes a while
# chrom_paths = glob.glob("/scratch/unified/hgdp_tgp_sgdp_high_cov_ancients_*.dated.trees.tsz")
# with ProcessPoolExecutor() as p:
#     filtered = dict(p.map(filtered_het_matrix, chrom_paths))

# +
# pickle.dump(filtered, gzip.open("/scratch/unified/adna_filtered.pkl.gz", "wb"))
# -

filtered = pickle.load(gzip.open("/scratch/unified/adna_filtered.pkl.gz", "rb"))

import eastbay as eb
from eastbay.data import RawContig

import numpy as np
results = {}
# skip afansievo for now
for i in range(4, 8):
    data = {
        k: RawContig(het_matrix=v[i : i + 2], afs=np.ones(1), window_size=100)
        for k, v in filtered.items()
    }
    test_data = data.pop(
        "/scratch/unified/hgdp_tgp_sgdp_high_cov_ancients_chr1_p.dated.trees.tsz"
    )
    train_data = list(data.values())
    pop = ancient_pops[i - 4].metadata["name"]
    results[pop] = eb.fit(train_data, test_data, mutation_rate=1.29e-8)

# oops
new_results = {p.metadata['name']: v for p, v in zip(ancient_pops[1:], results.values())}

# !mkdir -p /scratch/eastbay_paper/eb_estimates/adna_filtered

pickle.dump(new_results, open("/scratch/eastbay_paper/eb_estimates/adna_filtered/adna_filtered.pkl", "wb"))

# +
import matplotlib as mpl
import numpy as np
mpl.style.use("seaborn-v0_8-white")
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

T = np.geomspace(1e1, 1e5, 1000)

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

def plot_posterior(dms, ax, band, **kw):
    Nes = np.array([dm.eta(T, Ne=True) for dm in dms])
    q025, m, q975 = np.quantile(Nes, [0.025, 0.5, 0.975], axis=0)
    ax.plot(T, m, **kw)
    kw.pop('label')
    if band:
        ax.fill_between(T, q025, q975, **kw, alpha=.2)

for pop, ax in zip(ancient_pops[1:], axs.reshape(-1)):
    unfiltered = pickle.load(open(f'/scratch/eastbay_paper/eb_estimates/unified/{pop.id}/dms.pkl', "rb"))
    name = pop.metadata['name']
    filtered = new_results[name]
    plot_posterior(filtered, ax, True, label="Filtered")
    plot_posterior(unfiltered, ax, False, linestyle="--", label="Unfiltered")
    ax.set_title(name, y=.87)
    for d in ['top', 'right']:
        ax.spines[d].set_visible(False)
    ax.set_yscale('log')
    ax.set_xscale('log')
    axs[0, 0].legend()

for ax in axs.reshape(-1):
    ax.tick_params(which='major', width=1, length=3)

for ax in axs[1]:
    ax.set_xticks([10 ** i for i in range(1, 6)])
    ax.set_xlabel('Relative time (generations)')
    
for ax in axs[:, 0]:
    ax.set_ylabel("$N_e$")
    
fig.suptitle("Sensitivity to model misspecification")
fig.tight_layout()
fig.savefig("adna_filtered.pdf")
# -



