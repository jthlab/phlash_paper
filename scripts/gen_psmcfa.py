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
    wc = snakemake.wildcards
    nodes = tuple([int(i) for i in [wc.node1, wc.node2]])
    ts = tskit.load(snakemake.input[0])
    with open(snakemake.output[0], "w") as out:
        tup = (wc.chrom,) + nodes
        gen_psmcfa(ts, "chr%s_%d_%d" % tup, nodes, out, w=100)
