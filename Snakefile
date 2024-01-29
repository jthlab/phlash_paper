import json
import pickle
from functools import partial
import cyvcf2

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

ALL_OUTPUT = []

def input_for_all(_):
    return ALL_OUTPUT

rule all:
    input: input_for_all

for mod in 'unified', 'ccr', 'sim', 'phlash', 'psmc', 'smcpp', 'h2h', 'bottleneck':
    include: f'snakefiles/{mod}'
