from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from patchOTDA.datasets import MMS_DATA
import patchOTDA.domainAdapt as pOTDA
from ot.datasets import make_2D_samples_gauss, make_data_classif
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
import pytest


pOTDA.TIMEOUT = 5  # Disable timeout for testing, we know these test (probably) should not run forever


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

def test_tune_real_data():
    """Test tuning with real data."""
    da = pOTDA.PatchClampOTDA()

    # Load data
    Xs = MMS_DATA['CTKE_M1']['ephys']
    Xt = MMS_DATA['VISp_Viewer']['ephys']
    print(MMS_DATA.keys())
    #make sure the features are the same
    Xs = Xs.loc[:,MMS_DATA['joint_feats']].to_numpy()
    Xt = Xt.loc[:,MMS_DATA['joint_feats']].to_numpy()
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    Xs = imputer.fit_transform(Xs)
    Xt = imputer.transform(Xt)
    Xs = scaler.fit_transform(Xs)
    Xt = scaler.transform(Xt)

    da_tuned = pOTDA.PatchClampOTDA(flexible_transporter=True)
    da_tuned.tune(Xs, Xt, n_iter=100, n_jobs=2, method='unidirectional', verbose=True)

    # After tuning, fit and transform with the best parameters
    Xs_tuned = da_tuned.fit_transform(Xs, Xt)


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

def test_tune_timeout():
    """Test that tuning respects the timeout."""
    p = pOTDA.PatchClampOTDA(flexible_transporter=True)

    Xs = make_2D_samples_gauss(n=100, m=(0, 0), sigma=np.array([[1, 0], [0, 1]]))
    Xt = make_2D_samples_gauss(n=150, m=(2.5, .5), sigma=np.array([[1, -.8], [-.8, 1]]))

    # Set a very short timeout to force a timeout error
    pOTDA.TIMEOUT = 0.001

    #Should return penalty value
    p.tune(Xs=Xs, Xt=Xt, n_jobs=2, n_iter=4, method="unidirectional", verbose=True)


if __name__=="__main__":
    test_tune_real_data()