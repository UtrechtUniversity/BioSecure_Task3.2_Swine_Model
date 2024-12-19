# Model of a Farrow-to-Finish Swine Farm

This repository contains code and data for the farrow-to-finish swine farm model. The main components include the farrow-to-finish farm model Emulsion configuration file, a Python add-on script for farmer and animal movement simulation, and a Jupyter notebook for parameter estimation of the transmission parameters using Approximate Bayesian Computation with Sequential Monte Carlo simulation (ABC-SMC).

## Repository Structure

- `data/`: Folder containing the data used in the simulations.
- `LICENSE`: License file for the repository.
- `movement_12fatpens_outrans.py`: Python script add-on for simulating movements in the swine model.
- `parameter_estimation_notebook.ipynb`: Jupyter notebook for parameter estimation using ABC.
- `swine_model_weekly-12fatpens-FINAL.yaml`: EMULSION file for the swine model.
- `swine_model_weekly-12fatpens-weight-FINAL-swIAVs.yaml`: EMULSION file for the swine model (swIAVs version).
- `movement_12fatpens_swIAVs.py`: Python script add-on for simulating movements in the swine model (swIAVs version).

## Getting Started

### Prerequisites

Ensure you have the following packages installed:
- Python 3.x
- `emulsion==1.2rc5`
- `numpy`
- `pandas`
- `matplotlib`
- `pyabc`
- `docopt`
- `shutil`
- `uuid`
- `datetime`
- `csv`
- `dateutil`
- `random`

You can install the required packages using pip:
```bash
pip install numpy pandas matplotlib pyabc docopt python-dateutil emulsion==1.2rc5 betapert
