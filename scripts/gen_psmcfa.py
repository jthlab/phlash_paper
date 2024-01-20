from collections import defaultdict
from typing import TextIO

import cyvcf2
import textwrap
import tskit
import tszip
import gzip


def gen_psmcfa(
    var_iter,
    contig: str,
    out: TextIO,
    w: int = 100,
):
    "Generate a PSMCFA file for nodes in tree seqeuence."
    d = defaultdict(lambda: "T")
    for v in var_iter:
        if v['het']:
            d[int(v['position'] / w)] = "K"
    outstr = [d[i] for i in range(max(d) + 1)]
    print("> %s" % contig, file=out)
    print("\n".join(textwrap.wrap("".join(outstr), width=79)), file=out)
    print("", file=out)

def vcf_iter(vcf_file, sample):
    vcf = cyvcf2.VCF(vcf_file, gts012=True, samples=[sample])
    for v in vcf:
        d = {'position': v.POS}
        assert len(v.gt_types) == 1
        assert v.gt_types[0] in [0, 1, 2, 3]
        d['het'] = v.gt_types[0] == 1
        yield d


if __name__ == "__main__":
    wc = snakemake.wildcards
    # nodes = tuple([int(i) for i in [wc.node1, wc.node2]])
    # ts = tszip.decompress(snakemake.input[0])
    assert snakemake.output[0].endswith(".gz")
    contig = f"chr{wc.chrom}_sample{wc.sample}"
    with gzip.open(snakemake.output[0], mode="wt") as out:
        vi = vcf_iter(snakemake.input[0], wc.sample)
        gen_psmcfa(vi, contig, out, w=100)
