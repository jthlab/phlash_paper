# ---
# jupyter:
#   jupytext:
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

import pandas as pd
import numpy as np
import sys
import tempfile
import shutil

import pickle
try:
    df = pickle.load(open(snakemake.input[0], 'rb'))
except:
    df = pickle.load(open(sys.argv[1], 'rb'))

df = df.drop(columns=['eld', 'old', 'esfs', 'obsfs'])
df['l2'] = df['l2'].astype(float)

df['method'] = pd.Categorical(df['method'], categories=['smcpp', 'msmc2', 'phlash', 'fitcoal'])

df_mean_std = df.melt(id_vars=['model', 'method', 'n', 'rep'], var_name='metric').reset_index().set_index(
    ['model', 'n', 'metric'])
df_mean_std.sort_index(inplace=True)
df_mean_std.loc[('Africa_1T12', 1, 'l2')]
df_mean_std['value'] = df_mean_std['value'].astype(float)

try:
    outfile = snakemake.output[0]
except:
    outfile = sys.argv[2]


SE = outfile.endswith("_sd.tex")

if SE:
    sts = ['sem']
else:
    sts = ['mean']
summary_df = df.groupby(['model', 'n', 'method']).agg({k: sts for k in ['l2', 'tv1', 'sfs', 'ld']})

s2 = summary_df.pivot_table(values=['l2', 'tv1', 'sfs', 'ld'], index=['model', 'n'], columns='method')

s2.columns = s2.columns.set_names(['metric', 'stat', 'method'])



# +
from scipy.stats import ttest_ind

def f(x):
    ret = []
    for k, v in x.items():
        n = k[1]
        if n == 1000:
            ret.append(False)
            continue
        dist = df_mean_std.loc[k[:3]]
        method = k[-1]
        b = dist['method'] == method
        z = dist['value'].to_numpy()
        t = ttest_ind(z[b], z[~b], alternative="less")
        ret.append(t.pvalue < 0.05)
    return pd.Series(ret, index=x.index)
t_df = s2.stack(['metric', 'stat', 'method']).groupby(['model', 'n', 'stat', 'metric']).transform(f)
t_df


# +
def f(r):
    return r <= r.dropna().min()
    
min_df = s2.stack(['metric', 'stat']).apply(f, axis=1)
# -

min_df.index


# +
def f(x):
    ret = []
    metric, stat, method = x.name
    for k, v in x.items():
        model, n = k
        if np.isnan(v):
            ret.append("---")
        else:
            s = f"{v:.3g}"
            if "e" in s:  # scientific notation was chosen
                s = f"{v:.5f}"
            b = min_df.loc[(model, n, metric, stat)][method]
            if b and (not SE):
                s = r'\mathbf{' + s + '}'
                if t_df.loc[k + x.name]:
                    s += "^*"
            ret.append("$" + s + "$")
    return pd.Series(ret, index=x.index)

s3 = s2.apply(f)
s3.index = s3.index.set_levels([[str(x).replace('_', '\\_') for x in level] for level in s3.index.levels])
s3.index = s3.index.set_levels([[r"\texttt{" + x + "}" for x in level] for level in s3.index.levels])
# -

s3 = s3.stack(level="stat")
s3.index = s3.index.droplevel(-1).set_names(['Model', '$n$'])

s3 = s3.rename(columns={'phlash': r'\textsc{phlash}', 'smcpp': r"\textsc{SMC}\texttt{{++}}", 'msmc2': r'\textsc{MSMC2}', 'fitcoal': r'\textsc{FitCoal}'})


import subprocess
try:
    k = snakemake.wildcards.metric.split("_")[0]
except:
    k = "sfs"
err = dict([("l2", "$L^2$"), ("tv1", "total variation"), ("sfs", r"SFS"), ("ld", "LD")])[k]
s4 = s3.loc[:, (k,)]

if k in ("sfs", "ld"):
    s4 = s4[s4.index.get_level_values('$n$') != r'\texttt{1}']

s4.columns.name = None
    
if SE:
    cap = f"Standard errors for {err} error."
else:
    cap = f"Average {err} error."
if err == "l2":
    cap += " (Errors have been divided by $10^8$.)"
latex_str = s4.style.to_latex(hrules=True, clines="skip-last;data", environment="footnotesize")
with open(outfile, "wt") as f:
    f.write(latex_str)
latex_document = f"""
\\documentclass{{article}}
\\usepackage[margin=1in,letterpaper]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\begin{{document}}
{latex_str}
\\end{{document}}
"""
# td = tempfile.TemporaryDirectory()
# p = td.name + f"/table_{k}.tex"
p = outfile[:-3] + "full.tex"
with open(p, "wt") as f:
    f.write(latex_document)
# subprocess.run(["pdflatex", p])
# shutil.copyfile(td.name + f"/table_{k}.pdf", snakemake.output[0][:-4] + ".pdf")
