#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # prevent jax from using gpu


# In[2]:


BASE = "/scratch/eastbay_paper/eb_estimates"


# In[3]:


import tszip

pops = list(
    tszip.decompress(
        "/scratch/unified/hgdp_tgp_sgdp_high_cov_ancients_chr22_q.dated.trees.tsz"
    ).populations()
)


# In[5]:


import pickle
import numpy as np
from eastbay.util import tree_stack
from jax import jit, vmap
import numpy as np
import jax.numpy as jnp
import tqdm.auto as tqdm

t = np.geomspace(1e1, 1e5, 1000)


@jit
def Ne(dms):
    a = vmap(lambda eta: 1 / 2 / eta(t))(dms.eta)
    return jnp.median(a, 0)


res = {}
for i in tqdm.trange(len(pops)):
    f = f"{BASE}/unified/{i}/dms.pkl"
    dms = tree_stack(pickle.load(open(f, "rb")))
    res[i] = Ne(dms)


# In[7]:


from eastbay.liveplot import style_axis
import matplotlib.pyplot as plt
for i in res:
    plt.plot(t, res[i], color="black", alpha=.05)
style_axis(plt.gca())
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e1, 1e5)


# In[78]:


import json

pop_data = {}
for pop in pops:
    d = json.loads(pop.metadata)
    label = d.get("region") or d.get("super_population")
    if label in ("AFR", "AFRICA"):
        label = "Africa"
    if label in ("Europe", "EUR", "WestEurasia", "EUROPE"):
        label = "Europe"
    if label in ("EAS", "EastAsia", "EAST_ASIA"):
        label = "East Asia"
    if label == "OCEANIA":
        label = "Oceania"
    if label in ("America", "AMERICA", "AMR"):
        label = "America"
    if label in ("SAS", "SouthAsia"):
        label = "South Asia"
    if label in ("CentralAsiaSiberia", "CENTRAL_SOUTH_ASIA", "South Asia"):
        label = "Central/South Asia"
    pop_data[pop.id] = dict(name=d["name"], label=label)


# In[199]:


import pandas as pd

X = np.array(list(res.values()))
df = (
    pd.DataFrame(data=X.T, index=t)
    .melt(var_name="pop", value_name="y", ignore_index=False)
    .reset_index()
).rename({"x": "index"})
labels = {i: pop_data[i]["label"] for i in res}
px.line(
    data_frame=df,
    x="index",
    y="y",
    line_group="pop",
    log_x=True,
    log_y=True,
    labels=labels,
)


# In[200]:


get_ipython().run_line_magic('pinfo', 'px.line')


# In[152]:


df["Balochi"]


# In[90]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(with_std=False), PCA(3))

X = np.array(list(res.values()))
Xd = pipe.fit_transform(X)


# In[91]:


import plotly.express as px

px.scatter(
    x=Xd[:, 0],
    y=Xd[:, 1],
    color=list([v["label"] for v in pop_data.values()]),
)


# In[39]:


for y in cl.cluster_centers_:
    plt.plot(t, y)
plt.xscale('log')
plt.yscale('log')

