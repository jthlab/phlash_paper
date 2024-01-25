import tskit
import numpy as np
import re

def convert_scrm(snakemake):
    wc = snakemake.wildcards
    two_pop = "::" in wc.population
    with open(snakemake.input[0], "rt") as scrm_out: 
        cmd_line = next(scrm_out).strip()
        L = int(re.search(r'-r [\d.]+ (\d+)', cmd_line)[1])
        header = ["##fileformat=VCFv4.0", """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">"""]
        contig = wc.chrom
        header.append(f"##contig=<ID={contig},length={L}>")
        h = "#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT".split()
        n = int(wc.num_samples)
        h += ["sample%d" % i for i in range(n)]
        header.append("\t".join(h))
        while not next(scrm_out).startswith("position"):
            continue
        with open(snakemake.output[0], "wt") as vcf:
            print("\n".join(header), file=vcf)
            for line in scrm_out:
                if line.startswith("SFS: "):
                    with open(snakemake.output[1], 'wt') as sfs:
                        sfs.write(line[5:])
                    continue
                pos, _, *gts = line.strip().split(" ")
                pos = int(1 + float(pos))  # vcf is 1-based; if a variant has pos=0 it messes up bcftools
                cols = [contig, str(pos), ".", "A", "C", ".", "PASS", ".", "GT"]
                if two_pop:
                    # if there are two populations then we will combine haplotypes from each population
                    n = len(gts)
                    assert n % 2 == 0
                    gtz = zip(gts[:n // 2], gts[n // 2:])
                else:
                    gtz = zip(gts[::2], gts[1::2])
                cols += ["|".join(gt) for gt in gtz]
                print("\t".join(cols), file=vcf)


def convert_ts(snakemake):
    ts = tskit.load(snakemake.input[0])
    afs = ts.allele_frequency_spectrum(span_normalise=False, polarised=True)[1:-1]
    with open(snakemake.output[1], "wt") as out:
        out.write(" ".join(map(str, afs)))
    n = int(snakemake.wildcards.num_samples)
    with open(snakemake.output[0], "wt") as out:
        def xform(coord):
            return (1 + np.array(coord)).astype(int)
        ts.write_vcf(
            out,
            contig_id=snakemake.wildcards.chrom,
            individual_names=['sample%d' % i for i in range(n)],
            position_transform=xform,
        )


if __name__ == "__main__":
    if snakemake.input[0].endswith(".ts"):
        convert_ts(snakemake)
    else:
        assert snakemake.input[0].endswith(".scrm_out.txt")
        convert_scrm(snakemake)

