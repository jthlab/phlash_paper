import numpy as np
import pickle
import cyvcf2
import eastbay as eb
import eastbay.mcmc
import tskit
import tszip

contigs = []
breakpoint()
for f in snakemake.input:
    if f.endswith("tsz"):
        ts = tszip.decompress(f)
        nodes = [tuple(i.nodes) for i in ts.individuals()]
        if np.shape(nodes)[1] == 1: 
            # all the individuals are haploid, so we group them pairwise
            nodes = [tuple(row) for row in np.reshape(nodes, (-1, 2))]
        contigs.append(eb.contig(ts, nodes))
    else:
        assert f.endswith("bcf.csi")
        f = f[:-4]
        vcf = cyvcf2.VCF(f)
        assert len(vcf.seqlens) == 1
        L = vcf.seqlens[0]
        chrom = vcf.seqnames[0]
        region = f"{chrom}:1-{L}"
        samples = vcf.samples
        contigs.append(eb.contig(f, region=region, samples=samples))

test_data, *train_data = sorted(contigs, key=lambda c: c.L)
res = eb.fit(train_data, test_data, mutation_rate=snakemake.params.mutation_rate)
with open(snakemake.output[0], "wb") as f:
    pickle.dump(res, f)
