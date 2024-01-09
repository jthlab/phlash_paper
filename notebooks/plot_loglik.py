# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pickle
import pandas as pd
records = []
for res in snakemake.input:
    d = pickle.load(open(res, "rb"))
    for k in d:
        records.append({"overlap": k, "rel_err": d[k]})

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
