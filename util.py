import re
import demes
import numpy as np
import stdpopsim
from functools import cache
from eastbay.size_history import DemographicModel, SizeHistory


def get_chroms(species_name):
    species = stdpopsim.get_species(species_name)
    return [
        chrom.id
        for chrom in species.genome.chromosomes
        if chrom.ploidy == 2
        and re.match(r"\d+", chrom.id)
        and chrom.recombination_rate > 0
    ]


def get_default_mutation_rate(species_name):
    return stdpopsim.get_species(species_name).genome.mean_mutation_rate


def get_genome_length(species_name):
    species = stdpopsim.get_species(species_name)
    return sum(species.get_contig(chrom).length for chrom in get_chroms(species_name))

@cache
def get_truth(species_name, demographic_model, population):
    species = stdpopsim.get_species(species_name)
    mu = get_default_mutation_rate(species_name)
    if demographic_model == "Constant":
        t = np.array([0.0])
        Ne = np.array([species.population_size])
    else:
        model = species.get_demographic_model(demographic_model)
        md = model.model.debug()
        t_min = 10.
        t_max = 2 * md.epochs[-1].start_time + 1
        assert np.isinf(md.epochs[-1].end_time)
        t = np.r_[0., np.geomspace(t_min, t_max, 1000)]
        pop_index = [p.name for p in model.populations].index(population)
        c, _ = md.coalescence_rate_trajectory(t, {population: 2})
    eta = SizeHistory(t=t, c=c)
    true_dm = DemographicModel(eta=eta, theta=mu, rho=None)
    return true_dm


def ts_input_for_species(wc):
    return [
        f"simulations/{wc.seed}/{wc.species}/{wc.other_params}/chr%s.trees.tsz" % chrom
        for chrom in get_chroms(wc.species)
    ]


def scrm_stdpopsim_cmd(species_name, demographic_model, population, chrom, sample_size, seed, length_multiplier=1.0):
    species = stdpopsim.get_species(species_name)
    N0 = species.population_size
    if demographic_model == "Constant":
        dm = stdpopsim.PiecewiseConstantSize(N0)
    else:
        dm = species.get_demographic_model(demographic_model)
    chrom = species.get_contig(chrom)
    assert chrom.interval_list[0].shape == (1, 2)
    assert chrom.interval_list[0][0, 0] == 0.0
    L = chrom.interval_list[0][0, 1] * length_multiplier
    theta = 4 * N0 * chrom.mutation_rate * L
    assert chrom.recombination_map.rate.shape == (1,)
    rho = 4 * N0 * chrom.recombination_map.rate[0] * L
    g = dm.model.to_demes()
    samples = [0] * len(g.demes)
    i = [d.name for d in g.demes].index(population)
    n = 2 * sample_size
    samples[i] = n
    cmd = demes.to_ms(g, N0=N0, samples=samples)
    args = " ".join(
        map(
            str,
            ["-t", theta, "-r", rho, L, "--transpose-segsites", "-SC", "abs", "-p", 14, "-oSFS", "-seed", seed],
        )
    )
    return f"{n} 1 {cmd} {args}"


@cache
def get_N0(species_name, demographic_model, population):
    species = stdpopsim.get_species(species_name)
    dm = species.get_demographic_model(demographic_model)
    # get effective N0 for this population by mean coalescence time
    N0 = dm.model.debug().mean_coalescence_time({population: 2}) / 2
    return N0


def chrom_input(wc):
    species_name = wc.species
    chrom_name = wc.chrom
    # decide whether to use msp or scrm based on scaled recombination rate.
    species = stdpopsim.get_species(species_name)
    # get effective N0 for this population by mean coalescence time
    N0 = get_N0(species_name, wc.demographic_model, wc.population)
    chrom = species.get_contig(chrom_name)
    assert len(chrom.interval_list) == 1
    assert chrom.interval_list[0].shape == (1, 2)
    assert chrom.interval_list[0][0, 0] == 0.
    L = chrom.interval_list[0][0, 1]
    assert chrom.recombination_map.rate.shape == (1,)
    rho = chrom.recombination_map.rate[0]
    r = 4 * N0 * rho * L * LENGTH_MULTIPLIER
    if r > 1e5:
        logger.info(f"{species_name=} {chrom_name=} {r=} using scrm")
        # msp simulations with r>100000 take forever.
        ext = "scrm_out.txt"
    else:
        ext = "ts"
    return f"simulations/{wc.seed}/{wc.species}/{wc.demographic_model}/{wc.population}/n{wc.num_samples}/chr%s.%s" % (chrom_name, ext)

if __name__ == "__main__":
    cmd = scrm_stdpopsim_cmd("HomSap", "OutOfAfrica_2T12", "EUR", "chr1", 10, 1)
    print(cmd)
