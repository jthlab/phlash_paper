from dataclasses import replace
import pickle
import jax
import phlash
import phlash.fit.seq
import sys
import cyvcf2
import os
from concurrent.futures import ThreadPoolExecutor


def region_for_vcf(chrom_path):
    vcf = cyvcf2.VCF(chrom_path)
    assert len(vcf.seqnames) == 1
    chrom = vcf.seqnames[0]
    assert len(vcf.seqlens) == 1
    L = vcf.seqlens[0]
    return f"{chrom}:1-{L}"


def process_chrom(conf_entry):
    assert conf_entry.endswith(".pkl")
    c = pickle.load(open(conf_entry, 'rb'))
    c = c._replace(ld=None)
    return c

if __name__ == "__main__":
    # assert jax.local_devices()[0].platform == "gpu"
    conf = pickle.load(open(sys.argv[1], "rb"))
    test_data = process_chrom(conf["test_data"])
    try:
        num_workers = int(os.environ.get('SLURM_JOB_CPUS_PER_NODE'))
    except:
        num_workers = None
    with ThreadPoolExecutor(num_workers) as pool:
        train_data = list(pool.map(process_chrom, conf["train_data"]))
    res = phlash.fit.seq.fit(
        data=train_data,
        test_data=test_data,
        mutation_rate=conf["mutation_rate"],
        num_workers=num_workers,
        **conf.get('options', {})
    )
    pickle.dump(res, open(sys.argv[2], "wb"))
