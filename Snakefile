import json
import pickle
from functools import partial
import cyvcf2
import os
import pandas as pd

config["human_mutation_rate"] = 1.29e-8
os.environ["SCRM_PATH"] = config["scrm_path"]
os.environ["PSMC_PATH"] = config["psmc_path"]

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import scienceplots
plt.style.use('science')
mpl.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath} 
\usepackage{amssymb}
\usepackage{times}
'''


def load_file(path):
    if path.endswith("json"):
        decoder = json
        mode = "t"
    else:
        assert path.endswith("pkl"), path
        decoder = pickle
        mode = "b"
    with open(path, f"r{mode}") as f:
        return decoder.load(f)


def dump_file(obj, path):
    if path.endswith("json"):
        encoder = json
        mode = "t"
    else:
        assert path.endswith("pkl")
        encoder = pickle
        mode = "b"
    with open(path, f"w{mode}") as f:
        encoder.dump(obj, f)


workdir: config["workdir"]


ALL_OUTPUT = []


def input_for_all(_):
    return ALL_OUTPUT


rule all:
    input:
        input_for_all,


for mod in [
    "unified",
    "ccr",
    "sim",
    "phlash",
    "psmc",
    "smcpp",
    "h2h",
    "bottleneck",
    "composite",
    "fitcoal",
    "msmc2",
    "independence",
    "adna",
    "misc_plots",
    "bench"
]:

    include: f"snakefiles/{mod}"
