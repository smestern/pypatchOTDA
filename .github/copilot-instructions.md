# patchOTDA – Copilot Instructions

## Project Overview
patchOTDA is a Python package for integrating patch-clamp electrophysiology datasets using **Optimal Transport (OT) domain adaptation**. It wraps the [POT](https://pythonot.github.io/) library and provides automated hyperparameter tuning via [nevergrad](https://github.com/facebookresearch/nevergrad). The primary use case is aligning experimental datasets that suffer from batch effects (different labs, temperatures, solutions, etc.).

## Architecture

- **`patchOTDA/domainAdapt.py`** – Core module. Contains `PatchClampOTDA`, which extends sklearn's `BaseEstimator` and POT's `BaseTransport`. Follows `fit` / `transform` / `fit_transform` conventions. Also contains the `tune()` system (nevergrad-based hyperparameter optimization with joblib parallelism), error/distance functions (`gw_dist`, `rf_clf_dist`, etc.), and the `unbalancedFUGWTransporter` class.
- **`patchOTDA/domain.py`** – Modified fork of `ot.da` with torch backend support and custom sinkhorn solvers (`sinkhorn_lpl1_mm`, `sinkhorn_l1l2_gl`). ~2600 lines; avoid unnecessary edits.
- **`patchOTDA/datasets.py`** – Loads bundled `mms_data.pkl` (Allen Institute reference data). Access via `MMS_DATA` dict with keys like `'CTKE_M1'`, `'VISp_Viewer'`, `'joint_feats'`.
- **`patchOTDA/loadNWB.py`** – NWB (Neurodata Without Borders) electrophysiology file loader using h5py. Returns `(dataX, dataY, dataC, dt)` arrays.
- **`patchOTDA/external/skada.py`** – Wrapper around `skada` (scikit-adaptation) methods. Classes like `JDOT`, `JDOTC`, `EntropicOT` follow the same `fit`/`transform` pattern. Optional dependency.
- **`patchOTDA/nn/uniOTtab.py`** – Neural-network-based DA (`UniOTtab`) adapting the UniOT method for tabular data. Marked unstable. Uses PyTorch.
- **`patchOTDA/nn/uniood/`** – Vendored/modified UniOT codebase (models, trainers, dataloaders). Treat as internal dependency.

## Key Patterns

### sklearn-style API
All transporters follow `fit(Xs, Xt)` → `transform(Xs, Xt)`. The convention uses **`Xs`** (source) and **`Xt`** (target) with optional labels **`Ys`**, **`Yt`**. POT uses lowercase `ys`/`yt` internally; `PatchClampOTDA.fit()` maps `Ys→ys`, `Yt→yt`.

### Transporter selection
`PatchClampOTDA` accepts a transporter as a string (resolved via `getattr(ot.da, name)`) or a class. Default is `ot.da.EMDLaplaceTransport`. The `flexible_transporter=True` flag allows the tuner to search across multiple OT methods.

### Hyperparameter tuning
`tune()` supports two methods:
- `'unidirectional'` – Transports Xs→Xt and scores with an error function (default: `gw_dist` unsupervised, `rf_clf_dist` supervised).
- `'bidirectional'` – Round-trip Xt→Xs→Xt reconstruction error. Requires `flexible_transporter=True`.

Tuning uses `nevergrad.optimizers.Portfolio` with `joblib.Parallel` (threaded). A `@timeout()` decorator kills slow evaluations after `TIMEOUT` seconds (default 120s). Set `pOTDA.TIMEOUT = None` to disable in tests.

### Kwargs filtering
`getValidKwargs(func, argsDict)` introspects function signatures to pass only valid kwargs. `getOptimizableKwargs(func)` builds nevergrad parameter dicts from `DEFAULT_OPTIMIZABLE_KWARGS` matched by name.

### Optional dependencies
torch, unbalancedgw, skada, and h5py are imported with try/except and log warnings on failure. Guard new optional imports the same way.

## Build & Install
```bash
pip install git+https://github.com/smestern/pypatchOTDA.git
# Optional extras:
pip install git+https://github.com/scikit-adaptation/skada
pip install unbalancedgw
```
Build system: setuptools via `pyproject.toml`. Core deps: `POT`, `numpy`, `scipy`, `matplotlib`, `nevergrad`.

## Testing
Tests live in `tests/` and use pytest-compatible functions (no classes). Run with:
```bash
pytest tests/
```
- `test_tune.py` – Tests `PatchClampOTDA.tune()` with synthetic Gaussian data from `ot.datasets`.
- `test_external.py` – Tests `skada` wrappers and `UniOTtab`. Requires a local Excel file path (hardcoded).

Set `pOTDA.TIMEOUT = None` before long-running tune tests to avoid premature termination.

## Conventions
- Use `logging` module (not print) for diagnostic output; the module-level `logger` is already configured.
- Error functions for tuning take signature `(Xs, Xt, Ys, Yt) → float`. Return `9e5` for degenerate solutions (all NaN/zero).
- Data should be scaled (e.g., `StandardScaler`) and imputed before transport; OT solvers are sensitive to scale.
- The `nn` subpackage is explicitly unstable — changes there should be conservative.
