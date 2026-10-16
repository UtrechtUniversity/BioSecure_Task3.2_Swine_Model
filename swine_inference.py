"""Local simulation and observation handling for the estimation notebook."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parent
MODELS = {
    "default": REPOSITORY / "swine_model_weekly-12fatpens-FINAL.yaml",
    "swiavs": REPOSITORY / "swine_model_weekly-12fatpens-weight-FINAL-swIAVs.yaml",
}
PARAMETER_MAPS = {"default": {
    "trans": "transmission_rate",
    "trans_betwn_herds": "between_herd_trans",
    "trans_neighbor": "neighboring_herd_trans",
}}
PARAMETER_MAPS["swiavs"] = {**PARAMETER_MAPS["default"], "trans_aux": "third_path_trans"}


def simulate(parameters, *, steps=8, seed=None, model_variant="default"):
    """Return the model's I-state pen-count summary at weeks 0 through ``steps``.

    Each call uses an independent temporary output directory. An explicit seed
    initializes both Python and NumPy randomness; their caller states are
    restored afterward so simulation calls do not reset pyABC's random stream.
    EMULSION uses object-identity hashes for agents, so a fixed seed alone does
    not guarantee identical trajectories across calls or processes.
    This adapter is intended for the notebook's serial SingleCoreSampler.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if model_variant not in MODELS:
        raise ValueError(f"Unknown model variant: {model_variant}")
    parameter_map = PARAMETER_MAPS[model_variant]
    if set(parameters) != set(parameter_map):
        raise ValueError(f"Expected parameter names: {', '.join(parameter_map)}")
    rates = {name: float(value) for name, value in parameters.items()}
    if any(not np.isfinite(value) or value < 0 for value in rates.values()):
        raise ValueError("Transition weights must be finite and nonnegative")
    if seed is None:
        seed = int(np.random.randint(0, 2**32 - 1))
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")

    from docopt import docopt
    import emulsion.__main__ as emulsion_cli

    numpy_state, python_state = np.random.get_state(), random.getstate()
    python_path = sys.path[:]
    log = io.StringIO()
    try:
        np.random.seed(int(seed))
        random.seed(int(seed))
        with tempfile.TemporaryDirectory(prefix="swine-inference-") as directory:
            output = Path(directory)
            argv = [
                "run", str(MODELS[model_variant]), "--runs", "1", "--time", str(steps),
                "--seed", str(seed), "--quiet",
                "--code-path", str(REPOSITORY),
                "--input-dir", str(REPOSITORY / "data"),
                "--output-dir", str(output / "counts"),
                "--figure-dir", str(output / "figures"),
            ]
            for name, value in rates.items():
                argv.extend(["--param", f"{parameter_map[name]}={value}"])
            args = docopt(emulsion_cli.__doc__, argv=argv)
            try:
                with redirect_stdout(log):
                    emulsion_cli.main(args)
            except SystemExit as error:
                raise RuntimeError(f"EMULSION exited during simulation:\n{log.getvalue()}") from error
            counts_file = output / "counts" / "counts.csv"
            if not counts_file.is_file():
                raise RuntimeError(f"EMULSION did not produce counts:\n{log.getvalue()}")
            counts = pd.read_csv(counts_file)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        sys.path[:] = python_path

    keys = counts[["simu_id", "step", "population_id"]]
    if (len(counts) != (steps + 1) * 18 or keys.duplicated().any()
            or set(counts["step"]) != set(range(steps + 1))
            or set(counts["population_id"]) != set(range(18))
            or set(counts["simu_id"]) != {0}):
        raise RuntimeError("Simulation output is incomplete or contains duplicate observations")
    trajectory = counts.loc[counts["population_id"].eq(4)].sort_values("step")
    values = trajectory["nb_of_sampled_I_Jf"].to_numpy(dtype=float, copy=True)
    # The summary process has not run at week 0. Derive its initial value from
    # the initial I-state agents rather than silently filling missing data.
    if np.isnan(values[0]):
        initial_pens = counts.loc[
            counts["step"].eq(0) & counts["population_id"].between(4, 15)
        ]
        values[0] = int(initial_pens["I"].gt(0).sum())
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 12)).all():
        raise RuntimeError("Simulation produced invalid I-state partition counts")
    return values


def load_observations(path, *, steps):
    """Read CSV columns week,nb_I: one integer I-state partition count for every week 0..steps.

    ``nb_I`` counts pen partitions with an I-state agent (0..12), not agents.
    Missing, duplicate, nonfinite, fractional or out-of-range data are rejected.
    """
    data = pd.read_csv(path)
    if not {"week", "nb_I"}.issubset(data.columns):
        raise ValueError("Observation CSV must contain columns week and nb_I")
    weeks = pd.to_numeric(data["week"], errors="raise").to_numpy(dtype=float)
    values = pd.to_numeric(data["nb_I"], errors="raise").to_numpy(dtype=float)
    if (len(data) != steps + 1 or not np.isfinite(weeks).all()
            or set(weeks) != set(range(steps + 1))):
        raise ValueError(f"Observations must contain each week 0 through {steps} exactly once")
    if (not np.isfinite(values).all() or not ((0 <= values) & (values <= 12)).all()
            or not (values == np.floor(values)).all()):
        raise ValueError("nb_I must contain integer I-state partition counts between 0 and 12")
    return values[np.argsort(weeks)]


def distance(simulated, observed):
    """L1 distance between equally sized I-state partition-count trajectories."""
    left = np.asarray(simulated["infected_pens"], dtype=float)
    right = np.asarray(observed["infected_pens"], dtype=float)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Simulated and observed trajectories must be equally sized vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Trajectories must contain only finite values")
    return float(np.abs(left - right).sum())
