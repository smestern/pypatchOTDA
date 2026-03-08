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


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada_mapping_adapters():
    """Test mapping/alignment adapters: CORAL, OTMapping, LinearOT, EntropicOT."""
    Xs, Ys = make_data_classif(dataset="3gauss", n=150, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=150, nz=0.5)
    Xt[:, 0] += 3
    Xt[:, 1] += 3

    for name, cls in [
        ("CORALDA", skada.CORALDA),
        ("OTMapping", skada.OTMapping),
        ("LinearOT", skada.LinearOT),
        ("EntropicOT", skada.EntropicOT),
    ]:
        adapter = cls()
        adapter.fit(Xs, Xt, ys=Ys, yt=Yt)
        transformed = adapter.transform(Xs, Xt=Xt)
        assert transformed.shape == Xs.shape, (
            f"{name}: shape mismatch {transformed.shape} vs {Xs.shape}"
        )
        assert not np.all(np.isnan(transformed)), f"{name}: all NaN output"

        # fit_transform should produce same shape
        transformed2 = cls().fit_transform(Xs, Xt=Xt, ys=Ys, yt=Yt)
        assert transformed2.shape == Xs.shape, f"{name}: fit_transform shape mismatch"


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada_subspace_adapters():
    """Test subspace adapters: TCA, SubspaceAlignment, etc."""
    Xs, Ys = make_data_classif(dataset="3gauss", n=150, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=150, nz=0.5)
    Xt[:, 0] += 3
    Xt[:, 1] += 3

    adapters_to_test = [("TCA", skada.TCA)]

    # Only test optional adapters if available
    if skada.SubspaceAlignmentAdapter is not None:
        adapters_to_test.append(("SubspaceAlignmentDA", skada.SubspaceAlignmentDA))
    if skada.TransferJointMatchingAdapter is not None:
        adapters_to_test.append(("TransferJointMatchingDA", skada.TransferJointMatchingDA))
    if skada.TransferSubspaceLearningAdapter is not None:
        adapters_to_test.append(("TransferSubspaceLearningDA", skada.TransferSubspaceLearningDA))

    for name, cls in adapters_to_test:
        adapter = cls()
        adapter.fit(Xs, Xt, ys=Ys, yt=Yt)
        transformed = adapter.transform(Xs, Xt=Xt)
        # Subspace methods may reduce dimensionality
        assert transformed.shape[0] == Xs.shape[0], (
            f"{name}: sample count mismatch {transformed.shape[0]} vs {Xs.shape[0]}"
        )
        assert not np.all(np.isnan(transformed)), f"{name}: all NaN output"


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada_predictor_only():
    """Test predictor-only methods raise NotImplementedError on transform."""
    Xs, Ys = make_data_classif(dataset="3gauss", n=150, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=150, nz=0.5)
    Xt[:, 0] += 3
    Xt[:, 1] += 3

    predictors_to_test = []
    if skada._DASVMClassifier is not None:
        predictors_to_test.append(("DASVMClassifierDA", skada.DASVMClassifierDA))
    if skada._OTLabelProp is not None:
        predictors_to_test.append(("OTLabelPropDA", skada.OTLabelPropDA))

    if not predictors_to_test:
        pytest.skip("No predictor-only skada methods available")

    for name, cls in predictors_to_test:
        adapter = cls()
        # fit may warn/fail on small 2D data – that's OK, we only test transform raises
        adapter.fit(Xs, Xt, ys=Ys, yt=Yt)
        with pytest.raises(NotImplementedError):
            adapter.transform(Xs)


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada_with_patchclampotda():
    """Test using a skada wrapper as transporter= in PatchClampOTDA."""
    from patchOTDA import PatchClampOTDA

    Xs, Ys = make_data_classif(dataset="3gauss", n=150, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=150, nz=0.5)
    Xt[:, 0] += 3
    Xt[:, 1] += 3

    # Test with class reference
    otda = PatchClampOTDA(transporter=skada.CORALDA)
    result = otda.fit_transform(Xs, Xt, Ys=Ys, Yt=Yt)
    assert result.shape == Xs.shape, f"Class ref shape mismatch: {result.shape}"

    # Test with string lookup
    otda2 = PatchClampOTDA(transporter='coral')
    result2 = otda2.fit_transform(Xs, Xt, Ys=Ys, Yt=Yt)
    assert result2.shape == Xs.shape, f"String lookup shape mismatch: {result2.shape}"


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada_methods_dict():
    """Verify METHODS dict is populated and all entries are valid."""
    assert len(skada.METHODS) > 0, "METHODS dict is empty"
    # All original methods should be present
    for key in ["jdot", "jdotc", "otmapping", "entropicOT", "classOT", "linearOT", "coral", "TCA"]:
        assert key in skada.METHODS, f"'{key}' missing from METHODS"

    # TRANSFORM_CAPABLE_METHODS should exclude predictor-only
    for name, cls in skada.TRANSFORM_CAPABLE_METHODS.items():
        assert not issubclass(cls, skada.baseSkadaPredictor), (
            f"'{name}' in TRANSFORM_CAPABLE_METHODS but is predictor-only"
        )


@pytest.mark.skipif(not _HAS_SKADA, reason="skada not installed")
def test_skada_log_attribute():
    """Verify log_ attribute is set after fit for tune() compatibility."""
    adapter = skada.CORALDA()
    Xs, Ys = make_data_classif(dataset="3gauss", n=100, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=100, nz=0.5)
    adapter.fit(Xs, Xt, ys=Ys, yt=Yt)
    assert hasattr(adapter, "log_"), "log_ attribute missing after fit"
    assert "warning" in adapter.log_, "log_ must contain 'warning' key"


@pytest.mark.skipif(
    not _HAS_SKADA or not skada._HAS_SKADA_METRICS,
    reason="skada or skada.metrics not installed",
)
def test_skada_metrics():
    """Test skada metric error functions return a float."""
    Xs, Ys = make_data_classif(dataset="3gauss", n=100, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=100, nz=0.5)
    Xt[:, 0] += 3
    Xt[:, 1] += 3

    for name, func in [
        ("prediction_entropy", skada.prediction_entropy_error),
        ("soft_neighborhood_density", skada.soft_neighborhood_density_error),
        ("circular_validation", skada.circular_validation_error),
    ]:
        result = func(Xs, Xt, Ys, Yt)
        assert isinstance(result, (float, np.floating)), (
            f"{name} returned {type(result)}, expected float"
        )


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

if __name__ == "__main__":
    pytest.main([__file__])