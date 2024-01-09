import os
import stdpopsim
import re
import tskit
import json
import pandas as pd
import numpy as np
import pickle
import shutil
import tszip

import warnings
warnings.filterwarnings("ignore")

from eastbay.size_history import DemographicModel, SizeHistory

# METHODS = ["smcpp", "fitcoal", "eastbay", "psmc", "psmc32"]
METHODS = ["eastbay", "psmc64", "smcpp"]
MODELS = [
    # ("AnaPla", "MallardBlackDuck_2L19", "Mallard"),
    ("AnoGam", "GabonAg1000G_1A17", "GAS"),
    ("AraTha", "SouthMiddleAtlas_1D17", "SouthMiddleAtlas"),
    ("AraTha", "African3Epoch_1H18", "SouthMiddleAtlas"),
    ("BosTau", "HolsteinFriesian_1M13", "Holstein_Friesian"),
    ("DroMel", "African3Epoch_1S16", "AFR"),
    ("DroMel", "OutOfAfrica_2L06", "AFR"),
    ("HomSap", "Africa_1T12", "AFR"), 
    ("HomSap", "Africa_1B08", "African_Americans"), 
    ("HomSap", "Zigzag_1S14", "generic"), 
    ("PanTro", "BonoboGhost_4K19", "bonobo"),
    ("PapAnu", "SinglePopSMCpp_1W22", "PAnubis_SNPRC"),
    ("PonAbe", "TwoSpecies_2L11", "Bornean"),
]

SAMPLE_SIZES = [1, 10, 100]
NUM_REPLICATES = 10
LENGTH_MULTIPLIER = 1.0
# BCFTOOLS_CMD = "~/.local/bin/bcftools"
BCFTOOLS_CMD = "/usr/bin/bcftools"
BASE = "/home/terhorst/Dropbox (University of Michigan)/eastbay/manuscript_repo/assets"
TURBO = "/home/jonth/turbo/jonth/eastbay_paper"
os.environ["PSMC_PATH"] = os.path.join(BASE, "lib", "psmc", "psmc")

wildcard_constraints:
    chrom=r"\w+",
    species=r"\w+",
    population=r"\w+",
    num_samples=r"[0-9]+",
    seed=r"[0-9]+",
    demographic_model=r"\w+",


# workdir: "/scratch/jonth_root/jonth0/jonth/eastbay_paper/pipeline"
workdir: "/scratch/eastbay_paper"

def get_chroms(species_name):
    species = stdpopsim.get_species(species_name)
    return [
        chrom.id
        for chrom in species.genome.chromosomes
        if chrom.ploidy == 2 and 
        chrom.id.lower() not in ("mt", "x", "y") and
        chrom.recombination_rate > 0
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
        t_min = md.epochs[0].end_time * .5
        t_max = 2 * md.epochs[-1].start_time + 1
        assert np.isinf(md.epochs[-1].end_time)
        t = np.geomspace(t_min, t_max, 1000)
        pop_index = [p.name for p in model.populations].index(population)
        Ne = md.population_size_trajectory(t)[:, pop_index]
    eta = SizeHistory(t=t, c=1.0 / 2.0 / Ne)
    true_dm = DemographicModel(eta=eta, theta=mu, rho=None)
    return true_dm


def ts_input_for_species(wc):
    return [
        f"simulations/{wc.seed}/{wc.species}/{wc.other_params}/chr%s.trees.tsz" % chrom
        for chrom in get_chroms(wc.species)
    ]

# rule plot_loglik:
#     input:
#         [f"methods/eastbay/loglik/{species}/{demographic_model}/{population}/relerr.dat"
#          for species, demographic_model, population in MODELS]
#     output:
#         BASE + "figures/loglik.pdf"
#     script: 
#         "notebooks/plot_loglik.py"


# rule loglik_relerr:
#     output:
#         r"methods/eastbay/loglik/{species}/{demographic_model}/{population}/relerr.dat"
#     script:
#         "notebooks/loglik.py"

# rule all:
#     input:
#         "figures/loglik.pdf"

rule all:
    input:
        # "methods/smcpp/output/1/HomSap/Constant/pop_0/n2/dm.pkl",
        # "methods/psmc/output/1/HomSap/Constant/pop_0/n1/psmc_out.txt",
        # "methods/eastbay/output/1/HomSap/Constant/pop_0/n2/eb.dat",
        [f"figures/{species}/{dm}/{pop}/n{i}/fig.pdf" for i in SAMPLE_SIZES 
         for (species, dm, pop) in MODELS]

# begin rules
# these rules are fast and don't need to be jobbed out to the cluster
localrules: smcpp_to_csv, fitcoal_truncate, fitcoal_to_dm, plot, smcpp_to_dm

rule compress_ts:
    input:
        "simulations/{simulation_params}/chr{chrom}.trees"
    output:
        "simulations/{simulation_params}/chr{chrom}.trees.tsz"
    shell:
        "tszip {input}"


rule run_stdpopsim:
    output:
        temporary("simulations/{seed}/{species}/{demographic_model}/{population}/chr{chrom}.trees")
    resources:
        runtime=60,
        mem_mb=8000,
    run:
        template = (
            "stdpopsim "
            "--msprime-change-model 0 dtwf "
            "--msprime-change-model 100 hudson "
            "{wildcards.species} %s -c {wildcards.chrom} -o {output} "
            "-l %f -s {wildcards.seed} {wildcards.population}:%d "
        )
        if wildcards.demographic_model == "Constant":
            dm = ""
        else:
            dm = f" -d {wildcards.demographic_model} "
        shell(template % (dm, LENGTH_MULTIPLIER, max(SAMPLE_SIZES)))


rule gen_frequency_spectrum:
    input:
        ts_input_for_species,
    output:
        r"simulations/{seed}/{species}/{other_params}/n{num_samples}/afs.txt",
    run:
        afss = []
        for ts in map(tszip.decompress, input):
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
        "{path}.trees.tsz",
    output:
        temporary("{path}.vcf")
    run:
        # ensure that coordinates are "1-based"
        xform = lambda x: (1 + np.array(x)).astype(int)
        with open(output[0], "wt") as f:
            tszip.decompress(input[0]).write_vcf(f, position_transform=xform)

rule vcf2bcf:
    input: "{path}.vcf"
    output: "{path}.bcf"
    shell:
        "%s view -o {output[0]} {input[0]}" % BCFTOOLS_CMD


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
    threads: 8,
    resources:
        mem_mb=32000,
        runtime=1440,
    shell:
        "smc++ estimate --cores 8 --spline cubic --knots 8 --timepoints 1e2 1e5 "
        "-o {params.outdir} {params.mutation_rate} {input}"


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
            "methods/smcpp/output/{seed}/{species}/{other_params}/%s" % fn
            for fn in ["model.final.json", "plot.csv"]
        ],
    output:
        "methods/smcpp/output/{seed}/{species}/{other_params}/dm.pkl",
    run:
        df = pd.read_csv(input[1])
        eta = SizeHistory(t=df["x"].to_numpy(), c=1 / 2 / df["y"].to_numpy())
	mu = get_default_mutation_rate(wildcards.species)
        dm = DemographicModel(theta=mu, rho=None, eta=eta)
        with open(output[0], "wb") as f:
            pickle.dump(dm, f)


rule ts_to_psmcfa:
    input:
        "simulations/{other_params}/chr{chrom}.trees.tsz"
    output:
        temporary("simulations/{other_params}/n{num_samples}/chr{chrom}_{node1}_{node2}.psmcfa.gz")
    resources:
        runtime=120
    script:
        "scripts/gen_psmcfa.py"

def psmcfa_input_for_species(wc):
    n = int(wc.num_samples)
    return [
        r"simulations/{seed}/{species}/{other_params}/n{num_samples}/chr%s_%d_%d.psmcfa.gz" % (chrom, 2 * i, 2 * i + 1)
        for chrom in get_chroms(wc.species)
        for i in range(n)
    ]

rule combine_psmcfa:
    input:
        psmcfa_input_for_species,
    output: r"methods/psmc/input/{seed}/{species}/{other_params}/n{num_samples}/combined.psmcfa.gz"
    run:
        with open(output[0], "wt") as f:
            for fn in input:
                with open(fn, "rt") as fin:
                    shutil.copyfileobj(fin, f)


rule uncompress_gzip:
    input: "{file}.gz"
    output: temporary("{file}")
    shell: "gunzip {input[0]}"

rule psmc_estimate:
    input:
        r"methods/psmc/input/{seed}/{species}/{other_params}/n{num_samples}/combined.psmcfa"
    output:
        [
            r"methods/{method,psmc(64)?}/output/{seed}/{species}/{other_params}/n{num_samples}/%s"
            % fn
            for fn in ["dm.pkl", "psmc_out.txt"]
        ],
    resources:
        runtime=1440,
    script:
        "scripts/mspsmc.py"


rule fitcoal_truncate:
    input:
        r"simulations/{seed}/{species}/{other_params}/afs.txt",
    output:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/trunc.txt",
    resources:
        runtime=1440,
    shell:
        "java -cp %s/lib/FitCoal1.1/FitCoal.jar FitCoal.calculate.TruncateSFS "
        "-input {input} > {output}" % BASE

def get_fitcoal_trunc(wc, input):
    try:
        with open(input[1], "rt") as f:
            txt = f.readlines()
            m = re.match(r"The number of SFS types to be truncated or collapsed: (\d+) \(recommended\)", txt[4])
            assert m
            return int(m.group(1))
    except:
        # sometimes their tool just prints nothing, idk why. do no trunccation in this case.
        return 0

rule fitcoal_estimate:
    input:
        [r"simulations/{seed}/{species}/{other_params}/afs.txt",
         r"methods/fitcoal/output/{seed}/{species}/{other_params}/trunc.txt"]
    output:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/fitcoal.out.txt"
    params:
        mu=lambda wc: get_default_mutation_rate(wc.species) * 1000,  # mutation rate per kb per generation
        genome_length_kbp=lambda wc: int(get_genome_length(wc.species) / 1000),  # length of genome in kb
        output_base=lambda wc, output: os.path.splitext(output[0])[0],
        trunc=get_fitcoal_trunc,
    resources:
        runtime=1440,
    shell:
        "java -cp %s/lib/FitCoal1.1/FitCoal.jar FitCoal.calculate.SinglePopDecoder "
        "-table %s/lib/FitCoal1.1/tables/ "
        "-input {input[0]} -output {params.output_base} "
        "-mutationRate {params.mu} -generationTime 1 "
        "-omitEndSFS {params.trunc} "
        "-genomeLength {params.genome_length_kbp}" % (BASE, BASE)


rule fitcoal_to_dm:
    input:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/fitcoal.out.txt",
    output:
        r"methods/fitcoal/output/{seed}/{species}/{other_params}/dm.pkl",
    run:
        df = pd.read_csv(input[0], sep="\t")
        Ne = df["popSize"].to_numpy()
        eta = SizeHistory(t=df["year"].to_numpy(), c=1. / 2 / Ne)
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
        gpus=1,
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
        if k.startswith("psmc") and n > 1:
            continue
        ret[k] = [
                f"methods/{k}/output/{i}/{wc.species}/{wc.demographic_model}/{wc.population}/n{n}/dm.pkl"
                for i in range(NUM_REPLICATES)
                ]
    return ret

rule plot:
    input:
        unpack(input_for_plot)
    output:
        r"figures/{species}/{demographic_model}/{population}/n{num_samples}/fig.pdf"
    params:
        truth=lambda wc: get_truth(wc.species, wc.demographic_model, wc.population),
    script:
        "scripts/plot.py"
