import os
import stdpopsim
import tskit
import json
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

from eastbay.size_history import DemographicModel, SizeHistory

METHODS = ["smcpp", "psmc", "fitcoal", "eastbay"]
MAX_SAMPLE_SIZE = 10
NUM_REPLICATES = 1
LENGTH_MULTIPLIER = 0.01
BCFTOOLS_CMD = "~/.local/bin/bcftools"
BASE = "/home/jonth/eastbay_paper"
TURBO = "/home/jonth/turbo/jonth/eastbay_paper"
os.environ["PSMC_PATH"] = os.path.join(BASE, "lib", "psmc", "psmc")

wildcard_constraints:
    chrom=r"\w+",
    species=r"\w+",
    population=r"\w+",
    num_samples=r"[0-9]+",
    seed=r"[0-9]+",
    demographic_model=r"\w+",


workdir: "/scratch/jonth_root/jonth0/jonth/eastbay_paper/pipeline"
# workdir: "pipeline"


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


def get_truth(species_name, demographic_model, population):
    species = stdpopsim.get_species(species_name)
    mu = get_default_mutation_rate(species_name)
    if demographic_model == "Constant":
        t = np.array([0.0])
        Ne = np.array([species.population_size])
    else:
        model = species.get_demographic_model(demographic_model)
        md = model.model.debug()
        breakpoint()
        t_min = 1.0
        t_max = md.epochs[-1].start_time + 1
        assert np.isinf(md.epochs[-1].end_time)
        t = np.geomspace(t_min, t_max, 1000)
        Ne = md.population_size_trajectory(t)[:, pop_index]
    eta = SizeHistory(t=t, c=1.0 / 2.0 / Ne)
    true_dm = DemographicModel(eta=eta, theta=mu, rho=None)
    return true_dm


def ts_input_for_species(wc):
    return [
        f"simulations/{wc.seed}/{wc.species}/{wc.other_params}/chr%s.ts" % chrom
        for chrom in get_chroms(wc.species)
    ]


MODELS = [("HomSap", "Zigzag_1S14", "pop_0"), ("HomSap", "Constant", "pop_0")]


rule all:
    input:
        # "methods/smcpp/output/1/HomSap/Constant/pop_0/n2/dm.pkl",
        # "methods/psmc/output/1/HomSap/Constant/pop_0/n1/psmc_out.txt",
        # "methods/eastbay/output/1/HomSap/Constant/pop_0/n2/eb.dat",
        f"{TURBO}/figures/HomSap/Constant/pop_0/n1/fig.pdf",

# begin rules
# these rules are fast and don't need to be jobbed out to the cluster
localrules: smcpp_to_csv, combine_psmcfa, fitcoal_to_dm, plot

rule run_stdpopsim:
    output:
        "simulations/{seed}/{species}/{demographic_model}/{population}/chr{chrom}.ts",
    run:
        template = (
            "stdpopsim {wildcards.species} %s -c {wildcards.chrom} -o {output} "
            "-l %f -s {wildcards.seed} {wildcards.population}:{MAX_SAMPLE_SIZE}"
        )
        if wildcards.demographic_model == "Constant":
            dm = ""
        else:
            dm = f" -d {wildcards.demographic_model} "
        shell(template % (dm, LENGTH_MULTIPLIER))


rule gen_frequency_spectrum:
    input:
        ts_input_for_species,
    output:
        r"simulations/{seed}/{species}/{other_params}/n{num_samples}/afs.txt",
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
        "python3 -m tskit vcf {input} | %s view -o {output}" % BCFTOOLS_CMD


rule index_bcf:
    input:
        "{path}.bcf",
    output:
        "{path}.bcf.csi",
    shell:
        "%s index {input}" % BCFTOOLS_CMD


rule smcpp_vcf2smc:
    input:
        [
            r"simulations/{seed}/{species}/{other_params}/{chrom}.bcf" + ext
            for ext in ["", ".csi"]
        ],
    output:
        r"methods/smcpp/input/{seed}/{species}/{other_params}/n{sample_size}/{chrom}.smc.txt.gz",
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
        r"methods/smcpp/output/{seed}/{species}/{other_params}/model.final.json",
    params:
        outdir=lambda wc, output: os.path.dirname(output[0]),
        mutation_rate=lambda wc: get_default_mutation_rate(wc.species),
    threads: 8
    resources:
        mem_mb=32000,
        runtime=120
    shell:
        "smc++ estimate --cores 8 -o {params.outdir} {params.mutation_rate} {input}"


rule smcpp_to_csv:
    input:
        "methods/smcpp/output/{params}/model.final.json",
    output:
        ["methods/smcpp/output/{params}/plot.%s" % ext for ext in ["png", "csv"]],
    shell:
        "smc++ plot -c {output[0]} {input}"


rule smcpp_to_dm:
    input:
        [
            "methods/smcpp/output/{params}/%s" % fn
            for fn in ["model.final.json", "plot.csv"]
        ],
    output:
        "methods/smcpp/output/{params}/dm.pkl",
    run:
        mdl = json.load(open(input[0]))
        df = pd.read_csv(input[1])
        eta = SizeHistory(t=df["x"].to_numpy(), c=1 / 2 / df["y"].to_numpy())
        dm = DemographicModel(theta=mdl["theta"], rho=mdl["rho"], eta=eta)
        with open(output[0], "wb") as f:
            pickle.dump(dm, f)


rule ts_to_psmcfa:
    input:
        "{other_params}/chr{chrom}.ts"
    output:
        "{other_params}/n{num_samples}/chr{chrom}.psmcfa"
    script:
        "scripts/gen_psmcfa.py"

def psmcfa_input_for_species(wc):
    return [
        r"simulations/{seed}/{species}/{other_params}/n{num_samples}/chr%s.psmcfa" % chrom
        for chrom in get_chroms(wc.species)
    ]

rule combine_psmcfa:
    input:
        psmcfa_input_for_species,
    output: r"methods/psmc/input/{seed}/{species}/{other_params}/n{num_samples}/combined.psmcfa"
    shell:
        "cat {input} > {output}"

rule psmc_estimate:
    input:
        r"methods/psmc/input/{seed}/{species}/{other_params}/n{num_samples}/combined.psmcfa"
    output:
        [
            r"methods/psmc/output/{seed}/{species}/{other_params}/n{num_samples}/%s"
            % fn
            for fn in ["dm.pkl", "psmc_out.txt"]
        ],
    resources:
        runtime=120
    script:
        "scripts/mspsmc.py"

rule fitcoal_estimate:
    input:
        r"simulations/{seed}/{species}/{other_params}/afs.txt",
    output:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/fitcoal.out.txt",
    params:
        mu=lambda wc: get_default_mutation_rate(wc.species) * 1000,  # mutation rate per kb per generation
        genome_length_kbp=lambda wc: int(get_genome_length(wc.species) / 1000),  # length of genome in kb
        output_base=lambda wc, output: os.path.splitext(output[0])[0],
    resources:
        runtime=120
    shell:
        "java -cp %s/lib/FitCoal1.1/FitCoal.jar FitCoal.calculate.SinglePopDecoder "
        "-table %s/lib/FitCoal1.1/tables/ "
        "-input {input} -output {params.output_base} "
        "-mutationRate {params.mu} -generationTime 1 "
        "-genomeLength {params.genome_length_kbp}" % (BASE, BASE)


rule fitcoal_to_dm:
    input:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/fitcoal.out.txt",
    output:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/dm.pkl",
    run:
        df = pd.read_csv(input[0], sep="\t")
        eta = SizeHistory(t=df["year"].to_numpy(), c=df["popSize"].to_numpy())
        mu = get_default_mutation_rate(wildcards.species)
        dm = DemographicModel(theta=mu, rho=None, eta=eta)
        with open(output[0], "wb") as f:
            pickle.dump(dm, f)


rule eastbay_estimate:
    input:
        ts_input_for_species,
    output:
        r"methods/eastbay/output/{seed}/{species}/{other_params}/n{num_samples}/dm.pkl",
    resources:
        gpu=1,
        slurm_partition="spgpu",
        slurm_extra="--gpus 1",
        runtime=120,
        mem_mb=32000,
    script:
        "scripts/eb.py"

def input_for_plot(wc):
    n = int(wc.num_samples)
    ret = {}
    for k in METHODS:
        ret[k] = [
                f"methods/{k}/output/{i}/{wc.species}/{wc.demographic_model}/{wc.population}/n{wc.num_samples}/dm.pkl"
                for i in range(NUM_REPLICATES)
                ]
    return ret

rule plot:
    input:
        unpack(input_for_plot)
    output:
        r"%s/figures/{species}/{demographic_model}/{population}/n{num_samples}/fig.pdf" % TURBO
    params:
        truth=lambda wc: get_truth(wc.species, wc.demographic_model, wc.population),
    script:
        "scripts/plot.py"
