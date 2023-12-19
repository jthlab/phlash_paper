import eastbay.liveplot
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

if snakemake.wildcards.demographic_model == "Constant":
    t1 = 1e1
    tM = 1e5
else:
    t1 = truth.eta.t[1]
    tM = truth.eta.t[-1]

t = np.geomspace(0.5 * t1, 2 * tM, 1000)

truth = snakemake.params.truth
true_Ne = 1 / 2 / truth.eta(t)
ax.plot(t, true_Ne, color="black", label="Truth")

for method, fns in snakemake.input.items():
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
        ax.plot(t, Ne, label=method)

eastbay.liveplot.style_axis(ax)
ax.legend()
fig.savefig(snakemake.output[0])
