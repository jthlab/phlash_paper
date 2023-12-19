from typing import TextIO

import textwrap
import tskit
from tqdm.auto import tqdm


def gen_psmcfa(
    ts: tskit.TreeSequence,
    contig: str,
    nodes: tuple[int, int],
    out: TextIO,
    w: int = 100,
):
    "Generate a PSMCFA file for nodes in tree seqeuence."
    L = int(ts.get_sequence_length() // w)
    outstr = ["T"] * (L + 1)
    for v in tqdm(ts.variants(samples=nodes), total=ts.num_sites):
        gt = v.genotypes
        if gt[0] != gt[1]:
            outstr[int(v.position / w)] = "K"
    print("> %s" % contig, file=out)
    print("\n".join(textwrap.wrap("".join(outstr), width=79)), file=out)
    print("", file=out)


if __name__ == "__main__":
    n = int(snakemake.wildcards.num_samples)
    nodes = [(2 * i, 2 * i + 1) for i in range(n)]
    ts = tskit.load(snakemake.input[0])
    with open(snakemake.output[0], "w") as out:
        for i, h in enumerate(nodes):
            gen_psmcfa(ts, "contig%d" % i, h, out, w=100)
