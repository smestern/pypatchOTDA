import patchOTDA.domainAdapt as pOTDA
from ot.datasets import make_2D_samples_gauss, make_data_classif
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
import pytest


pOTDA.TIMEOUT = None  # Disable timeout for testing


def test_tune_unsuper():
    """Test unsupervised tuning with synthetic Gaussian data."""
    p = pOTDA.PatchClampOTDA(flexible_transporter=False)

    Xs = make_2D_samples_gauss(n=100, m=(0, 0), sigma=np.array([[1, 0], [0, 1]]))
    Xt = make_2D_samples_gauss(n=150, m=(2.5, .5), sigma=np.array([[1, -.8], [-.8, 1]]))

    p.tune(Xs=Xs, Xt=Xt, n_jobs=2, n_iter=4, method="unidirectional", verbose=True)

    Xs_shifted = p.fit_transform(Xs, Xt)

    # Basic shape checks
    assert Xs_shifted.shape == Xs.shape, f"Expected shape {Xs.shape}, got {Xs_shifted.shape}"
    # The shifted data should not be identical to the original
    assert not np.allclose(Xs_shifted, Xs), "Shifted data should differ from original"
    # The shifted data should not be all NaN or all zero (degenerate)
    assert not np.all(np.isnan(Xs_shifted)), "Shifted data should not be all NaN"
    assert not np.all(Xs_shifted == 0), "Shifted data should not be all zeros"
    # The shifted data should be closer to Xt than Xs was (in terms of mean)
    original_dist = np.linalg.norm(Xs.mean(axis=0) - Xt.mean(axis=0))
    shifted_dist = np.linalg.norm(Xs_shifted.mean(axis=0) - Xt.mean(axis=0))
    assert shifted_dist < original_dist, (
        f"Shifted data mean should be closer to target: original_dist={original_dist:.3f}, shifted_dist={shifted_dist:.3f}"
    )


def test_tune_supervised():
    """Test supervised tuning with labelled synthetic data."""
    Xs, Ys = make_data_classif(dataset="3gauss", n=100, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=150, nz=0.5)

    p = pOTDA.PatchClampOTDA('SinkhornLpl1Transport', flexible_transporter=False)

    p.tune(Xs=Xs, Xt=Xt, Ys=Ys, Yt=Yt, n_jobs=2, n_iter=2, method="unidirectional", supervised=True, verbose=True)

    Xs_shifted = p.fit_transform(Xs, Xt, Ys, Yt)

    # Basic shape and sanity checks
    assert Xs_shifted.shape == Xs.shape, f"Expected shape {Xs.shape}, got {Xs_shifted.shape}"
    assert not np.all(np.isnan(Xs_shifted)), "Shifted data should not be all NaN"
    assert not np.all(Xs_shifted == 0), "Shifted data should not be all zeros"
    assert hasattr(p, 'best_'), "Tuned model should have best_ attribute"
