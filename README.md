# Agent-Based Modelling of *Aspergillus niger* Pellet Morphology

Code accompanying a study that extracts morphological information from confocal Z-stack imaging of *Aspergillus niger* mycelium and uses it to build, calibrate, and analyse an agent-based model (ABM) of pellet growth under different nutrient conditions.

## Overview

The repository covers the full pipeline from raw microscopy data to a validated, computationally cheap surrogate model:

```
data_extraction              Raw Z-stack images  →  per-hypha morphology (length, volume, branching)
        │
        ▼
collective_exports           Experimental growth curves per medium (fitted trends)
        │
        ▼
mycelium_model_iteration     Agent-based model of hyphal growth, branching & nutrient uptake
        │
        ▼
sensitivity_analysis /       Parameter, seed & nutrient sensitivity sweeps
RNG_seed_and_nutrient_
sweep_sensitivity_results
        │
        ▼
surrogate_model               ML surrogate of the ABM + MCMC Bayesian calibration
```

Three growth media/conditions are studied throughout: **MM** (Minimal Media), **MDM** (Mn-Deficient Medium), and **MSM** (Mn-Sufficient Medium).

## Folder Guide

### `data_extraction/`
Raw confocal/STEVE Z-stack images (as PNG slices, organised by medium and frame) plus `master_plotter_manualv2.py`, the extraction pipeline. For each frame it thresholds the image stack into a 3D point cloud, lets the user manually assign points to individual hyphae via an interactive 3D picker, then skeletonizes each hypha (graph-Laplacian contraction, adaptive resampling, MST-based graph reconstruction) to compute real-world (µm) length, volume, tip counts, and branch events. Outputs per-hypha and global-metric CSVs and time-series plots used to calibrate and validate the ABM. Also includes `branching_tracking_MSM.csv`, manually tracked ground-truth branching events.

### `collective_exports/`
Aggregated experimental growth curves derived from `data_extraction`: average length and volume per hypha over time for each medium, together with fitted exponential-saturation curves (`y = a·exp(bx) + c`) and their R² values. Used as the experimental target data for model comparison and calibration.

### `mycelium_model_iteration/`
The agent-based model itself. Each simulation represents hyphal tips growing through a 3D nutrient field (nitrogen, phosphorus, glucose, oxygen), with:
- conidial (spore) seeding with minimum spacing,
- Michaelis–Menten nutrient uptake and diffusion (explicit-Euler, optionally parallelised),
- apical and lateral branching rules driven by local nutrient availability,
- citric acid production once nitrogen/phosphorus are depleted.

Includes medium-specific variants (`MM_diff3.py`, `MDM_diff3.py`, `MSM_diff3.py` in `working models/`), full-diffusion and parallelised pellet versions (`myc_model4pellet.py`, `parallel_pellet_diff3_*.py`), and `sensitivity_analysis_MDM_sim.py`, which exposes the MDM model as a callable `run_simulation(overrides=None)` function used by the sensitivity and surrogate-modelling stages. `older_models/` contains earlier, superseded iterations kept for reference.

### `sensitivity_analysis/` and `RNG_seed_and_nutrient_sweep_sensitivity_results/`
Outputs of sensitivity studies run on the MDM model: RNG-seed sweeps (stochastic noise floor), ±% nutrient sweeps, and one-at-a-time ±5% parameter perturbations. `sensitivity_analysis/figures/` contains tornado plots ranking each parameter's effect on hyphal length, volume, citric acid production, tip count, and branching events. These CSVs feed directly into the MCMC calibration in `surrogate_model/`.

### `surrogate_model/`
Machine-learning surrogate of the ABM and Bayesian parameter calibration:
- `surrogate_model_testing.ipynb` trains and compares five regressors (linear, SVR, tuned SVR, random forest, gradient boosting, Gaussian process) mapping 12 ABM input parameters to 5 output metrics (average length, average volume, tip count, apical/lateral branching events). Gradient boosting gives the best validation performance (R² 0.80–0.92).
- `model_MCMC.ipynb` uses the trained gradient-boosting surrogate as a fast forward model inside an `emcee` MCMC sampler to infer the posterior distribution of key ABM parameters given target outputs, producing corner plots, posterior summaries, and posterior-predictive validation.

`original_csv_files/` holds the raw parameter sweeps used to train the surrogate; `model_csv_files/` holds the cleaned train/validation split; `exports_and_results/` holds the resulting plots and posterior summaries.

## Dependencies

Python 3, with `numpy`, `scipy`, `matplotlib`, `networkx`, `scikit-learn`, `Pillow`, `emcee`, and `corner`.

## Suggested Usage Order

1. `data_extraction/master_plotter_manualv2.py` — extract morphology from Z-stacks (update `DATASET_DIR` to your local path).
2. Compare against `collective_exports/` growth curves.
3. Run a medium-specific model from `mycelium_model_iteration/working models/` or `mycelium_model_iteration/`.
4. Explore parameter sensitivity via `mycelium_model_iteration/sensitivity_analysis_MDM_sim.py` and the results in `sensitivity_analysis/`.
5. Train/inspect the surrogate and run MCMC calibration in `surrogate_model/`.
