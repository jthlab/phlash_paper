from collections import defaultdict
import sys
from typing import TextIO

import cyvcf2
import textwrap
import tskit
import tszip
import gzip


def gen_psmcfa(
    vi,
    contig: str,
    L: int,
    out: TextIO,
    w: int = 100,
):
    "Generate a PSMCFA file for nodes in tree seqeuence."
    d = ["T"] * int(L // w + 1)
    for v in vi:
        if v['het']:
            d[int(v['position'] / w)] = "K"
    print("> %s" % contig, file=out)
    print("\n".join(textwrap.wrap("".join(d), width=79)), file=out)
    print("", file=out)

def vcf_iter(vcf, sample):
    for v in vcf:
        d = {'position': v.POS}
        assert len(v.gt_types) == 1
        assert v.gt_types[0] in [0, 1, 2, 3]
        d['het'] = v.gt_types[0] == 1
        yield d


if __name__ == "__main__":
    try:
        wc = snakemake.wildcards
        chrom = wc.chrom
        sample = wc.sample
        output = snakemake.output[0]
        input_ = snakemake.input[0]
    except:
        sample, chrom, input_, output = sys.argv[1:]
    # nodes = tuple([int(i) for i in [wc.node1, wc.node2]])
    # ts = tszip.decompress(snakemake.input[0])
        assert output.endswith(".gz")
    contig = f"chr{chrom}_sample{sample}"
    vcf = cyvcf2.VCF(input_, gts012=True, samples=[sample])
    i = vcf.seqnames.index(chrom)
    L = vcf.seqlens[i]
    with gzip.open(output, mode="wt") as out:
        vi = vcf_iter(vcf, sample)
        gen_psmcfa(vi, contig, L, out, w=100)
