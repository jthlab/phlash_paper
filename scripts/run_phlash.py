import pickle
import jax
import phlash
import sys
import cyvcf2
import os

def region_for_vcf(chrom_path):
    vcf = cyvcf2.VCF(chrom_path)
    assert len(vcf.seqnames) == 1
    chrom = vcf.seqnames[0]
    assert len(vcf.seqlens) == 1
    L = vcf.seqlens[0]
    return f"{chrom}:1-{L}"


def process_chrom(conf_entry):
    path = conf_entry[0]
    if path.endswith(".tsz"):
        path, nodes = conf_entry
        return phlash.contig(path, nodes)
    elif path.endswith(".bcf"):
        path, samples = conf_entry
        region = region_for_vcf(path)
        return phlash.contig(path, samples=samples, region=region)
    elif path.endswith(".pkl"):
        contig = load_file(path)
        assert isinstance(contig, phlash.data.Contig)
        return contig
    else:
        raise ValueError("unknown file type")


if __name__ == "__main__":
    assert jax.local_devices()[0].platform == "gpu"
    conf = pickle.load(open(sys.argv[1], "rb"))
    test_data = process_chrom(conf["test_data"])
    train_data = list(map(process_chrom, conf["train_data"]))
    res = phlash.fit(
        data=train_data,
        test_data=test_data,
        mutation_rate=conf["mutation_rate"],
        fold_afs=False,
    )
    pickle.dump(res, open(sys.argv[2], "wb"))
