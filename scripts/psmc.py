"""Interface to command-line PSMC that takes tree sequence input."""
import pickle
import itertools
import logging
import os
import tempfile
import textwrap
from dataclasses import dataclass
from typing import List, TextIO, Tuple, Union

import numpy as np
import sh
import tskit
from eastbay.size_history import DemographicModel, SizeHistory
from tqdm.auto import tqdm

psmc = sh.Command(os.environ.get("PSMC_PATH", "psmc"))

__version__ = next(
    line.strip().split(" ")[1]
    for line in psmc(_err_to_out=True, _ok_code=1).split("\n")
    if line.startswith("Version")
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PSMCPosterior:
    t: np.ndarray
    gamma: np.ndarray


def _gen_psmcfa(
    ts: tskit.TreeSequence,
    contig: str,
    nodes: Tuple[int, int],
    out: TextIO,
    w: int = 100,
):
    "Generate a PSMCFA file for nodes in tree seqeuence."
    L = int(ts.get_sequence_length() // w)
    outstr = ["T"] * (L + 1)
    for v in tqdm(ts.variants(samples=nodes), total=ts.num_sites):
        gt = v.genotypes
        if gt[0] != gt[1]:
            outstr[int(v.position / w)] = "K"
    print("> %s" % contig, file=out)
    print("\n".join(textwrap.wrap("".join(outstr), width=79)), file=out)
    print("", file=out)


def _psmciter(out):
    "split psmc output on // and return groups of lines"
    i = 0

    def keyfunc(line):
        nonlocal i
        if line.startswith("//"):
            i += 1
        return i

    # find last estimate
    return [list(lines) for i, lines in itertools.groupby(out, keyfunc)]


def _parse_psmc(out) -> List[DemographicModel]:
    "Parse PSMC output"
    iterations = _psmciter(out)
    ret = []
    for iterate in iterations:
        d = []
        theta = rho = None
        for line in iterate:
            if line.startswith("TR"):
                theta, rho = list(map(float, line.strip().split("\t")[1:3]))
            elif line.startswith("RS"):
                d.append(list(map(float, line.strip().split("\t")[2:4])))
        if d:
            t, lam = np.transpose(d)
            ret.append(
                DemographicModel(
                    theta=theta, rho=rho, eta=SizeHistory(t=t, c=1.0 / lam)
                )
            )
    return ret


def _parse_posterior(out):
    "Parse PSMC posterior output"
    iterations = _psmciter(out)
    posterior = iterations[-1][1:]  # strip leading //
    groups = itertools.groupby(posterior, lambda line: line[:2])
    _, times = next(groups)
    t = [0.0] + [float(line.strip().split("\t")[4]) for line in times]
    _, pd = next(groups)
    gamma = np.array([list(map(float, line.strip().split("\t")[3:])) for line in pd])
    return PSMCPosterior(np.array(t), gamma)


@dataclass
class msPSMC:
    """PSMC model.

    Args:
         data: List of `(tree sequence, (hap1, hap2))` pairs.
         w: window size (default=100)
    """

    psmcfa: str
    w: int = 100

    def estimate(
        self, *args, timepoints=None, full_output=False
    ) -> Union[DemographicModel, List[DemographicModel]]:
        """Run psmc and store results."""
        with tempfile.NamedTemporaryFile(
            "wt", suffix=".psmc"
        ) as f, tempfile.NamedTemporaryFile("wt", suffix=".txt") as params:
            if timepoints is not None:
                out = psmc(*args, "-N", 0, self.psmcfa)
                pa = next(line.strip() for line in out if line.startswith("PA"))
                _, numbers = pa.split("\t")
                pattern, theta, rho, max_n, *lam = numbers.split(" ")
                tp = " ".join(map(str, timepoints))
                par = f"{pattern} {theta} {rho} -1 {' '.join(lam)} {tp}"
                print(par, file=params, flush=True)
                args += ("-i", params.name)
            psmc("-o", f.name, *args, self.psmcfa)
            f = open(f.name)
            ret = _parse_psmc(f)
            f.seek(0)
            self._log = f.read()
        return ret if full_output else ret[-1]

    def posterior(self, *args):
        """Return posterior decoding"""
        with tempfile.NamedTemporaryFile(suffix=".psmc") as f:
            psmc("-o", f.name, "-d", "-D", *args, self.psmcfa)
            with open(f.name) as f:
                res = _parse_psmc(f)
                f.seek(0)
                pd = _parse_posterior(f)
                f.seek(0)
                self._log = f.read()
        return {"etas": res, "posterior": pd}


if __name__ == "__main__":
    p = msPSMC(snakemake.input[1])
    args = snakemake.params.args
    dm = p.estimate(*args)
    # scale up mutation and recombination rates
    dm = dm._replace(theta=dm.theta / 100, rho=dm.rho / 100)
    dm = dm.rescale(snakemake.params.mutation_rate)
    with open(snakemake.output[0], "wb") as f:
        pickle.dump(dm, f)
