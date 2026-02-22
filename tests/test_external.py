"""Tests for external wrappers (skada, UniOTtab).

These tests use synthetic data. Tests that require optional dependencies
are skipped when those dependencies are not installed.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from ot.datasets import make_2D_samples_gauss, make_data_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import ot.da


# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------
try:
    from patchOTDA.external import skada
    _HAS_SKADA = True
except ImportError:
    _HAS_SKADA = False

try:
    from patchOTDA.nn import uniOTtab
    from patchOTDA.nn.uniood.configs import parser
    _HAS_UNIOT = True
except ImportError:
    _HAS_UNIOT = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_UNIOT, reason="torch / uniOTtab not installed")
def test_uniOTtab_model():
    """Train UniOTtab on synthetic data and verify predictions are reasonable."""
    Xs, Ys = make_data_classif(dataset="3gauss", n=500, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=500, nz=0.5)

    # Shift target domain
    Xt[:, 0] += 4
    Xt[:, 1] += 4

    model = uniOTtab.UniOTtab()
    model.fit(Xs, Xt, Ys, Yt, n_share=4, n_source_private=0,
              num_internal_cluster=4, max_iter=50, base_lr=1e-3)

    preds_target = model.predict(Xt)
    preds_source = model.predict(Xs)

    # Predictions should have the right shape
    assert preds_target.shape == Yt.shape, f"Target pred shape mismatch: {preds_target.shape} vs {Yt.shape}"
    assert preds_source.shape == Ys.shape, f"Source pred shape mismatch: {preds_source.shape} vs {Ys.shape}"

    # Source accuracy should be reasonable (> chance = 1/n_classes)
    source_acc = accuracy_score(Ys, preds_source)
    assert source_acc > 0.4, f"Source accuracy too low: {source_acc:.3f}"


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada():
    """Test JDOTC wrapper with synthetic data."""
    jdot = skada.JDOTC(n_iter_max=int(1e3))

    Xs, Ys = make_data_classif(dataset="3gauss", n=200, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=300, nz=0.5)

    # Shift target domain
    Xt[:, 0] += 4
    Xt[:, 1] += 4

    Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(Xs, Ys, test_size=0.6, random_state=42)
    Xt_train, Xt_test, Yt_train, Yt_test = train_test_split(Xt, Yt, test_size=0.6, random_state=42)

    jdot.fit(Xt_test, Xs_train, Yt_test, Ys_train)
    preds = jdot.predict(Xt)

    assert preds.shape == Yt.shape, f"Prediction shape mismatch: {preds.shape} vs {Yt.shape}"

    # Transform should return same number of samples
    transformed = jdot.transform(Xt_train)
    assert transformed.shape[0] == Xt_train.shape[0], "Transform should preserve sample count"


def test_ub_sink():
    """Test unbalanced Sinkhorn transport with simple shifted data."""
    OT = ot.da.UnbalancedSinkhornTransport()
    Xs = make_2D_samples_gauss(n=100, m=10, sigma=[[2, 1], [1, 2]], random_state=42)
    Xt = (Xs + 0.5).astype('float32')
    Xs = Xs.astype('float32')

    OT.fit(Xs=Xs, Xt=Xt)
    Xs_transformed = OT.transform(Xs=Xs, Xt=Xt)

    assert Xs_transformed.shape == Xs.shape, f"Shape mismatch: {Xs_transformed.shape} vs {Xs.shape}"
    assert not np.all(np.isnan(Xs_transformed)), "Transformed data should not be all NaN"
    # Transformed data should be closer to target
    orig_dist = np.linalg.norm(Xs.mean(axis=0) - Xt.mean(axis=0))
    trans_dist = np.linalg.norm(Xs_transformed.mean(axis=0) - Xt.mean(axis=0))
    assert trans_dist < orig_dist, "Transformed data should be closer to target"
