import json
import pickle
from functools import partial

def load_file(path):
    if path.endswith("json"):
        decoder = json
        mode = 't'
    else:
        assert path.endswith("pkl")
        decoder = pickle
        mode = 'b'
    with open(path, f'r{mode}') as f:
        return decoder.load(f)

def dump_file(obj, path):
    if path.endswith("json"):
        encoder = json
        mode = 't'
    else:
        assert path.endswith("pkl")
        encoder = pickle
        mode = 'b'
    with open(path, f'w{mode}') as f:
        encoder.dump(obj, f)

workdir: "/scratch/eastbay_paper/pipeline"

rule all:
    input:
        ["figures/ccr/plot.pdf", "figures/h2h/plot.pdf"][0]

for mod in 'unified', 'ccr', 'sim', 'phlash', 'psmc', 'h2h':
    include: f'snakefiles/{mod}'
