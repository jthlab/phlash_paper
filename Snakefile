import os
os.environ['PSMC_PATH'] = '/scratch/psmc/psmc'
import stdpopsim
import tskit
import numpy as np
MAX_SAMPLE_SIZE=100

def get_chroms(species):
    species = stdpopsim.get_species(species)
    return [chrom.id for chrom in species.genome.chromosomes 
            if chrom.ploidy == 2 and 
            chrom.id.lower() not in ("mt", "x", "y")]

def get_default_mutation_rate(species):
    return stdpopsim.get_species(species).mean_mutation_rate
    
MODELS = [
    ('HomSap', "Zigzag_1S14", "pop_0"),
    ('HomSap', "Constant", "pop_0")
]

rule all:
    input:
        "methods/smcpp/results/1/HomSap/Constant/pop_0/n2/model.final.json"

rule run_stdpopsim:
    output:
        "simulations/{seed}/{species}/{demographic_model}/{population}/chr{chrom}.ts"
    run:
        template = (
            "stdpopsim {wildcards.species} %s -c {wildcards.chrom} -o {output} "
            "-l 0.1 -s {wildcards.seed} {wildcards.population}:{MAX_SAMPLE_SIZE}"
        )
        if wildcards.demographic_model == 'Constant':
            dm = ""
        else:
            dm = f" -d {wildcards.demographic_model} "
        shell(template % dm)

def ts_input_for_inference_method(wildcards):
    base = f"simulations/{wildcards.species}/{wildcards.demographic_model}/{wildcards.seed}/"
    return [base + chrom + ".ts" for chrom in get_chroms(wildcards.species)]

def chrom(wildcards):
    return f"simulations/{wildcards.species}/{wildcards.demographic_model}/{wildcards.seed}/{wildcards.chrom}.ts"

rule gen_frequency_spectrum:
    input:
        "{path}.ts"
    output:
        "{path}.afs.txt"
    run:
        ts = tskit.load(input[0])
        n = int(wildcards.num_samples)
        afs = ts.allele_frequency_spectrum(
            sample_sets=[list(range(2 * n))], 
            polarised=True, 
            span_normalise=False)[1:-1].astype(int)
        with open(output[0], "wt") as f:
            f.write(" ".join(map(str, afs)))

rule ts2vcf:
    input:
        "{path}.ts"
    output:
        "{path}.bcf"
    shell:
        "python3 -m tskit vcf {input} | bcftools view -o {output}"

rule index_bcf:
    input:
        "{path}.bcf"
    output:
        "{path}.bcf.csi"
    shell:
        "bcftools index {input}"

rule smcpp_vcf2smc:
    input:
        ["simulations/{params}/{chrom}.bcf" + ext for ext in ["", ".csi"]]
    output:
        "methods/smcpp/data/{params}/n{sample_size}/{chrom}.smc.txt.gz"
    run:
        sample_ids = ",".join([f"tsk_{i}" for i in range(int(wildcards.sample_size))])
        pop_str = "pop1:" + sample_ids
        shell(f"smc++ vcf2smc {input[0]} {output} 1 {pop_str}")

def smcpp_input_for_estimate(wc):
    return ",".join([f"methods/smcpp/data/{wc.species}/{wc.other_params}/%s.smc.txt.gz" % chrom
            for chrom in get_chroms(species)])

rule smcpp_estimate:
    input:
        smcpp_input_for_estimate
    output:
        "methods/smcpp/results/{species}/{other_params}/model.final.json"
    params:
        outdir=lambda wc, output: os.path.dirname(output[0])
        mutation_rate=lambda wc: get_default_mutation_rate(wc.species)
    shell:
        "smc++ estimate -o {params.outdir} {params.mutation_rate} {input}"

rule fit_psmc:
    input:
        ts_input_for_inference_method
    output:
        "psmc/{species}/{demographic_model}/n{num_samples}/{seed}/psmc.dat",
        "psmc/{species}/{demographic_model}/n{num_samples}/{seed}/psmc.log.txt"
    script:
        "scripts/mspsmc.py"
