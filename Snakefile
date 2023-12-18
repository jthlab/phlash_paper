import os

os.environ["PSMC_PATH"] = "/scratch/psmc/psmc"
import stdpopsim
import tskit
import numpy as np

MAX_SAMPLE_SIZE = 100


def get_chroms(species_name):
    species = stdpopsim.get_species(species_name)
    return [
        chrom.id
        for chrom in species.genome.chromosomes
        if chrom.ploidy == 2 and chrom.id.lower() not in ("mt", "x", "y")
    ]


def get_default_mutation_rate(species_name):
    return stdpopsim.get_species(species_name).genome.mean_mutation_rate


def get_genome_length(species_name):
    species = stdpopsim.get_species(species_name)
    return sum(species.get_contig(chrom).length for chrom in get_chroms(species_name))


def ts_input_for_species(wc):
    return [
        f"simulations/{wc.seed}/{wc.species}/{wc.other_params}/chr%s.ts" % chrom
        for chrom in get_chroms(wc.species)
    ]


MODELS = [("HomSap", "Zigzag_1S14", "pop_0"), ("HomSap", "Constant", "pop_0")]


rule all:
    input:
        # "methods/smcpp/output/1/HomSap/Constant/pop_0/n2/model.final.json",
        # "methods/psmc/output/1/HomSap/Constant/pop_0/n2/psmc.out.txt",
        "methods/fitcoal/output/1/HomSap/Constant/pop_0/n2/fitcoal.out.txt",


rule run_stdpopsim:
    output:
        "simulations/{seed}/{species}/{demographic_model}/{population}/chr{chrom}.ts",
    run:
        template = (
            "stdpopsim {wildcards.species} %s -c {wildcards.chrom} -o {output} "
            "-l 0.1 -s {wildcards.seed} {wildcards.population}:{MAX_SAMPLE_SIZE}"
        )
        if wildcards.demographic_model == "Constant":
            dm = ""
        else:
            dm = f" -d {wildcards.demographic_model} "
        shell(template % dm)


rule gen_frequency_spectrum:
    input:
        ts_input_for_species,
    output:
        r"simulations/{seed,\d+}/{species,\w+}/{other_params}/n{num_samples,\d+}/afs.txt",
    run:
        afss = []
        for ts in map(tskit.load, input):
            ts = tskit.load(input[0])
            n = int(wildcards.num_samples)
            afs = ts.allele_frequency_spectrum(
                sample_sets=[list(range(2 * n))], polarised=True, span_normalise=False
            )[1:-1].astype(int)
            afss.append(afs)
        afs = np.sum(afss, 0)
        with open(output[0], "wt") as f:
            f.write(" ".join(map(str, afs)))


rule ts2vcf:
    input:
        "{path}.ts",
    output:
        "{path}.bcf",
    shell:
        "python3 -m tskit vcf {input} | bcftools view -o {output}"


rule index_bcf:
    input:
        "{path}.bcf",
    output:
        "{path}.bcf.csi",
    shell:
        "bcftools index {input}"


rule smcpp_vcf2smc:
    input:
        [
            r"simulations/{seed,\d+}/{species,\w+}/{other_params}/{chrom,\w+}.bcf"
            + ext
            for ext in ["", ".csi"]
        ],
    output:
        r"methods/smcpp/input/{seed}/{species}/{other_params}/n{sample_size,\d+}/{chrom,\w+}.smc.txt.gz",
    run:
        sample_ids = ",".join([f"tsk_{i}" for i in range(int(wildcards.sample_size))])
        pop_str = "pop1:" + sample_ids
        shell(f"smc++ vcf2smc {input[0]} {output} 1 {pop_str}")


def smcpp_input_for_estimate(wc):
    return [
        f"methods/smcpp/input/{wc.seed}/{wc.species}/{wc.other_params}/chr%s.smc.txt.gz"
        % chrom
        for chrom in get_chroms(wc.species)
    ]


rule smcpp_estimate:
    input:
        smcpp_input_for_estimate,
    output:
        r"methods/smcpp/output/{seed,\d+}/{species,\w+}/{other_params}/model.final.json",
    params:
        outdir=lambda wc, output: os.path.dirname(output[0]),
        mutation_rate=lambda wc: get_default_mutation_rate(wc.species),
    shell:
        "smc++ estimate -o {params.outdir} {params.mutation_rate} {input}"


rule psmc_estimate:
    input:
        ts_input_for_species,
    output:
        [
            r"methods/psmc/output/{seed,\d+}/{species,\w+}/{other_params}/n{num_samples,\d+}/psmc.%s"
            % ext
            for ext in ["dat", "out.txt"]
        ],
    script:
        "scripts/mspsmc.py"


rule fitcoal_estimate:
    input:
        r"simulations/{seed,\d+}/{species,\w+}/{other_params}/afs.txt",
    output:
        r"methods/fitcoal/output/{seed,\d+}/{species,\w+}/{other_params}/fitcoal.out.txt",
    params:
        mu=lambda wc: get_default_mutation_rate(wc.species),
        genome_length_kbp=lambda wc: int(get_genome_length(wc.species) / 1000),
        output_base=lambda wc, output: os.path.splitext(output[0])[0],
    shell:
        "java -cp lib/FitCoal1.1/FitCoal.jar FitCoal.calculate.SinglePopDecoder "
        "-table lib/FitCoal1.1/tables/ "
        "-input {input} -output {params.output_base} "
        "-mutationRate {params.mu} -generationTime 1 "
        "-genomeLength {params.genome_length_kbp}"
