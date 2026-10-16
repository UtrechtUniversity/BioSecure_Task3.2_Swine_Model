![EMULSION 1.2rc5](https://img.shields.io/badge/emulsion-1.2rc5-brightgreen) ![pyABC 0.12.15](https://img.shields.io/badge/pyABC-0.12.15-blue)

# Model of a Farrow-to-Finish Swine Farm

EMULSION configurations and Python add-ons model a farrow-to-finish swine farm.
The notebook demonstrates estimation of transition weights using Approximate
Bayesian Computation with Sequential Monte Carlo (ABC-SMC).

Related article: [DOI: 10.1016/j.atech.2025.101652](https://doi.org/10.1016/j.atech.2025.101652).
The notebook uses the swIAVs configuration for a small computational demonstration,
not a reproduction of the article's analysis. See [reproducibility notes](docs/reproducibility.md).

## Install

The tested environment is **CPython 3.12.14 on Linux**, with EMULSION 1.2rc5 and
pyABC 0.12.15. Use Python 3.12 for these instructions. Other Python versions and
operating systems have not been validated with the pinned dependency set.

```bash
git clone https://github.com/UtrechtUniversity/BioSecure_Task3.2_Swine_Model.git
cd BioSecure_Task3.2_Swine_Model
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell instead of `source`.
`requirements.txt` lists the direct dependencies; `requirements-lock.txt`
records the complete tested dependency set. Built-in modules such as `csv`,
`random`, `shutil`, and `datetime` do not need separate installation.

## Run the default model

Run commands from the repository directory. Start with a short simulation:

```bash
emulsion run --plot swine_model_weekly-12fatpens-FINAL.yaml \
  --runs 1 --time 8 --seed 123 \
  --output-dir outputs/smoke --figure-dir img/smoke
```

For a full 170-week simulation, use `--time 170`. Without `--runs 1`, EMULSION's
CLI defaults to 10 repetitions. Outputs include a `counts.csv` file and an HTML
plot. Select a different output directory to retain results from multiple runs.
Do not pass `--plot` for inference model evaluations.

The separate swIAVs configuration can be run with:

```bash
emulsion run --plot swine_model_weekly-12fatpens-weight-FINAL-swIAVs.yaml \
  --runs 1 --time 8 --seed 123 \
  --output-dir outputs/swiavs --figure-dir img/swiavs
```

These are different computational rule-set variants. Passing an execution check
does not establish mathematical agreement between variants or with the article.

## Run parameter estimation

From the activated environment in the repository directory:

```bash
python -m ipykernel install --sys-prefix --name python3 --display-name "HogDSim (Python 3.12)"
python -m jupyter lab
```

Open `parameter_estimation_notebook.ipynb`, select the HogDSim kernel, and choose
**Restart Kernel and Run All Cells**. The defaults run the swIAVs configuration
for 8 weekly steps, then a small serial ABC-SMC example with four transition
weights, 12 particles, and at most 2 populations. The validated default run took about 15 minutes in the cloud environment;
runtime varies by machine. No Redis service or cluster
is required.

The notebook uses **explicitly labelled synthetic observations** by default.
The original notebook's `infected_data.csv` is absent. The article itself used
synthetic state-count sequences; the new example does not claim to recover its
original sequences or published estimates. Its four generating weights and prior
ranges follow sections 2.3.1 and 2.3.3. Adaptive Manhattan distance is used by default.
Set `SYNTHETIC_REPLICATES=100` to generate candidates and select an actual sequence
closest to their componentwise median, as described in the article. The quick
default uses one candidate; a full 100-candidate analysis is not run automatically.

To use your observations, set `OBSERVATIONS_CSV` in the notebook to your CSV path.
Required columns are `week,nb_I`, with exactly one row per week from 0 through
`SIMULATION_STEPS`. `nb_I` is the integer count of pen partitions containing an agent in state `I`
(0 to 12), **not the number of individual I-state agents**. Missing, duplicate, fractional,
or out-of-range observations are rejected. Supplied observations are never
overwritten with synthetic data.

Each notebook execution saves its configuration and observation provenance,
observations and synthetic candidates, SQLite ABC history, posterior particles/weights, and plots in a
unique `results/estimation-*` directory. Restarting the notebook creates another
run instead of overwriting the previous one. Increase the trajectory length and
inference budget only after checking the observation definition, priors,
convergence, and weight identifiability. Seeds initialize random streams but do
not guarantee identical trajectories in EMULSION 1.2rc5; see the reproducibility
notes. Save the actual generated observations for reuse.

For a noninteractive run of **all** notebook cells with the same defaults:

```bash
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 \
  --output executed-estimation.ipynb --output-dir results \
  parameter_estimation_notebook.ipynb
```

Create `results` first if this is your first run: `mkdir results`.

## Checks

```bash
python -m unittest discover -s tests -v
```

The tests run the real default model across two short repetitions, check complete
outputs, life-stage herd assignments and population totals, exercise the
inference adapter outside the checkout, check valid repeated simulations and exact caller
random-state preservation, and reject invalid observation files and trajectory
shapes. The complete notebook execution above additionally exercises ABC-SMC,
SQLite storage, and posterior plots. These are software checks, not validation of the mathematical rule sets or
an exact reproduction of the article. Known equation/code differences are
documented for a separate review; transition-weight formulas are preserved here.

## Troubleshooting

EMULSION's `--code-path` defaults to the **working directory**, not the YAML's
parent directory. When running from elsewhere, pass absolute `--code-path` and
`--input-dir` paths pointing to this repository and its `data` directory.

If EMULSION reports `module not found for level metapop`, use the same Python
environment and run this from the repository directory:

```bash
python -c "import movement_12fatpens_outrans"
```

The full traceback identifies the underlying error. EMULSION 1.2rc5 can report
the same message for a missing dependency inside the add-on, such as `betapert`.
Install the pinned requirements into the environment running EMULSION.

If the traceback reports `NameError: name 'herds' is not defined` at module scope,
check whether you are using an older copy of `movement_12fatpens_outrans.py`.
The repaired version keeps all three processes inside `Metapopulation` and
restores the explicit herd assignments from the earlier implementation.

`SWINE_MODEL.yaml` was a placeholder in the older notebook. The current adapter
selects the shipped default or swIAVs YAML explicitly and resolves paths
relative to the adapter, rather than requiring a renamed YAML file.

## Repository contents

- `swine_model_weekly-12fatpens-FINAL.yaml`: default model.
- `movement_12fatpens_outrans.py`: default model movement and state-transition processes.
- `swine_model_weekly-12fatpens-weight-FINAL-swIAVs.yaml` and
  `movement_12fatpens_swIAVs.py`: swIAVs variant.
- `data/modified_movements_10yr.csv`: supplied schedule input; not generated state-count observations.
- `parameter_estimation_notebook.ipynb`: executable synthetic-data ABC-SMC example.
- `swine_inference.py`: tested simulation adapter, observation loader, and distance.
- `tests/`: regression and inference tests.
- `requirements.txt`, `requirements-lock.txt`: direct and resolved dependencies.

Further documentation: [EMULSION](https://sourcesup.renater.fr/www/emulsion-public/pages/Getting_started.html)
and [pyABC](https://pyabc.readthedocs.io/en/latest/).
