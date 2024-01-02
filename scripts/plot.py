import eastbay.liveplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from eastbay.size_history import DemographicModel, SizeHistory

fig, ax = plt.subplots(figsize=(8, 5))

METHOD_TITLES = {
    "fitcoal": "FitCoal",
    "smcpp": "SMC++",
    "psmc": "PSMC",
    "eastbay": "eastbay",
}

truth = snakemake.params.truth

if snakemake.wildcards.demographic_model == "Constant":
    t1 = 1e1
    tM = 1e5
else:
    t1 = truth.eta.t[1]
    tM = truth.eta.t[-1]

t = np.geomspace(0.5 * t1, 2 * tM, 1000)

true_Ne = 1 / 2 / truth.eta(t)
ax.plot(t, true_Ne, color="black", label="Truth")

palette = mpl.colormaps['Set1']

for i, (method, fns) in enumerate(snakemake.input.items()):
    Nes = []
    for fn in fns:
        with open(fn, "rb") as f:
            dm = pickle.load(f)
        if method == "eastbay":
            # list of posterior samples from dm, have to compute posterior median
            m = np.quantile([d.rescale(truth.theta).eta(t) for d in dm], 0.5, axis=0)
            eta = SizeHistory(t=t, c=m)
            dm = DemographicModel(eta=eta, theta=truth.theta, rho=None)
        c = dm.rescale(truth.theta).eta(t)
        Ne = 1 / 2 / c
        Nes.append(Ne)
    q025, m, q975 = np.quantile(Nes, [0.025, 0.5, 0.975], axis=0)
    col = palette(i)
    ax.plot(t, m, color=col, label=method)
    ax.fill_between(t, q025, q975, color=col, alpha=.1)

eastbay.liveplot.style_axis(ax)
ax.legend()
fig.savefig(snakemake.output[0])
