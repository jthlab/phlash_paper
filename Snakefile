import json
import pickle
from functools import partial
import cyvcf2
import os
import os.path
import pandas as pd

config["human_mutation_rate"] = 1.29e-8
os.environ["SCRM_PATH"] = config["scrm_path"]
os.environ["PSMC_PATH"] = config["psmc_path"]

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import scienceplots
plt.style.use('science')
mpl.rcParams['font.size'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'
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
    with open(path, "r" + mode) as f:
        return decoder.load(f)


def dump_file(obj, path):
    if path.endswith("json"):
        encoder = json
        mode = "t"
    else:
        assert path.endswith("pkl")
        encoder = pickle
        mode = "b"
    with open(path, "w" + mode) as f:
        encoder.dump(obj, f)


workdir: config["workdir"]


ALL_OUTPUT = []


def input_for_all(_):
    return ALL_OUTPUT


rule all:
    input:
        input_for_all,


include: "snakefiles/unified"
include: "snakefiles/ccr"
include: "snakefiles/sim"
include: "snakefiles/phlash"
include: "snakefiles/psmc"
include: "snakefiles/smcpp"
include: "snakefiles/h2h"
include: "snakefiles/bottleneck"
include: "snakefiles/composite"
include: "snakefiles/fitcoal"
include: "snakefiles/msmc2"
include: "snakefiles/independence"
include: "snakefiles/adna"
include: "snakefiles/misc_plots"
include: "snakefiles/bench"
