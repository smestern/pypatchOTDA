"""Wrappers around scikit-adaptation (skada) methods for the patchOTDA API.

All transform-capable classes follow the ``fit(Xs, Xt, Ys, Yt)`` /
``transform(Xs, Xt)`` pattern and are compatible with ``PatchClampOTDA``
as a ``transporter=`` argument (including ``tune()``).

Predictor-only classes (DASVM, OTLabelProp, etc.) support ``fit`` /
``predict`` but raise ``NotImplementedError`` from ``transform``.

Requires the ``skada`` package to be installed::

    pip install git+https://github.com/scikit-adaptation/skada

Wrapped methods are registered in the module-level ``METHODS`` dict so that
``PatchClampOTDA`` can resolve them by name string.
"""
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core skada imports (required)
# ---------------------------------------------------------------------------
try:
    from skada import (
        JDOTRegressor,
        JDOTClassifier,
        CORAL,
        CORALAdapter,
        TransferComponentAnalysisAdapter,
    )
    from skada import (
        OTMapping as _OTMapping,
        OTMappingAdapter,
        EntropicOTMappingAdapter,
        ClassRegularizerOTMappingAdapter,
        LinearOTMappingAdapter,
        make_da_pipeline,
    )
    from skada import source_target_split
except ImportError:
    logger.warning("skada not installed, external.skada wrappers will not work")
    raise

# ---------------------------------------------------------------------------
# Optional skada imports – guarded per-method so that older skada versions
# don't break the whole module.
# ---------------------------------------------------------------------------
try:
    from skada import SubspaceAlignmentAdapter
except ImportError:
    SubspaceAlignmentAdapter = None
    logger.debug("SubspaceAlignmentAdapter not available in this skada version")

try:
    from skada import TransferJointMatchingAdapter
except ImportError:
    TransferJointMatchingAdapter = None
    logger.debug("TransferJointMatchingAdapter not available in this skada version")

try:
    from skada import TransferSubspaceLearningAdapter
except ImportError:
    TransferSubspaceLearningAdapter = None
    logger.debug("TransferSubspaceLearningAdapter not available in this skada version")

try:
    from skada import MMDLSConSMappingAdapter
except ImportError:
    MMDLSConSMappingAdapter = None
    logger.debug("MMDLSConSMappingAdapter not available in this skada version")

try:
    from skada import MultiLinearMongeAlignmentAdapter
except ImportError:
    MultiLinearMongeAlignmentAdapter = None
    logger.debug("MultiLinearMongeAlignmentAdapter not available in this skada version")

try:
    from skada import DASVMClassifier as _DASVMClassifier
except ImportError:
    _DASVMClassifier = None
    logger.debug("DASVMClassifier not available in this skada version")

try:
    from skada import OTLabelProp as _OTLabelProp
except ImportError:
    _OTLabelProp = None
    logger.debug("OTLabelProp not available in this skada version")

try:
    from skada import JCPOTLabelProp as _JCPOTLabelProp
except ImportError:
    _JCPOTLabelProp = None
    logger.debug("JCPOTLabelProp not available in this skada version")

try:
    from skada import OTLabelPropAdapter as _OTLabelPropAdapter
except ImportError:
    _OTLabelPropAdapter = None
    logger.debug("OTLabelPropAdapter not available in this skada version")

try:
    from skada import JCPOTLabelPropAdapter as _JCPOTLabelPropAdapter
except ImportError:
    _JCPOTLabelPropAdapter = None
    logger.debug("JCPOTLabelPropAdapter not available in this skada version")

# ---------------------------------------------------------------------------
# Skada metrics (optional)
# ---------------------------------------------------------------------------
_HAS_SKADA_METRICS = False
try:
    from skada.metrics import (
        PredictionEntropyScorer,
        SoftNeighborhoodDensity,
        CircularValidation,
    )
    _HAS_SKADA_METRICS = True
except ImportError:
    logger.debug("skada.metrics scorers not available")

import numpy as np
from ot.backend import get_backend
from ot import dist
from sklearn.ensemble import RandomForestClassifier as _RFC

# ===================================================================
# METHODS dict – populated at module bottom after class definitions
# ===================================================================
METHODS = {}  # filled at end of module


# ===================================================================
# Base classes
# ===================================================================

class baseSkada:
    """Base class wrapping skada *Adapter* methods into the POT-compatible
    ``fit(Xs, Xt, ys, yt)`` / ``transform(Xs, Xt)`` / ``fit_transform``
    interface expected by ``PatchClampOTDA``.

    Class-level ``_adapter`` attribute must be set to the skada Adapter class.
    """

    _adapter = None  # override in subclass

    def __init__(self, **kwargs):
        if self._adapter is None:
            raise TypeError(
                f"{type(self).__name__}._adapter is None – the required "
                "skada class is not available in this installation"
            )
        self.model = self._adapter(**kwargs)
        self.kwargs = kwargs
        # POT-compatible log dict checked by _tune_transporter
        self.log_ = {"warning": None}

    # -- POT-compatible fit ---------------------------------------------------
    def fit(self, Xs, Xt, ys=None, yt=None, Ys=None, Yt=None, **kwargs):
        """Fit the adapter on source/target data.

        Accepts both POT-style lowercase ``ys``/``yt`` and patchOTDA-style
        uppercase ``Ys``/``Yt`` for convenience.
        """
        ys = ys if ys is not None else Ys
        yt = yt if yt is not None else Yt
        if ys is None:
            ys = np.zeros(Xs.shape[0])
        if yt is None:
            yt = np.zeros(Xt.shape[0])
        X = np.concatenate((Xs, Xt), axis=0)
        Y = np.concatenate((ys, yt), axis=0)
        domain = np.concatenate(
            (np.ones(Xs.shape[0]), -np.ones(Xt.shape[0])), axis=0
        )
        self.log_ = {"warning": None}
        try:
            self.model.fit(X, Y, sample_domain=domain, **kwargs)
        except Exception as e:
            self.log_["warning"] = str(e)
            logger.warning("skada fit failed: %s", e)
        self.xt_ = Xt
        self.xs_ = Xs
        return self

    # -- predict --------------------------------------------------------------
    def predict(self, Xs, **kwargs):
        return self.model.predict(Xs, **kwargs)

    # -- POT-compatible transform ---------------------------------------------
    def transform(self, Xs, Xt=None, ys=None, yt=None, Ys=None, Yt=None, **kwargs):
        """Transport ``Xs`` toward the target distribution.

        For JDOT-based methods the OT coupling is used directly;
        for other adapters the skada ``transform`` method is called.
        """
        if self.log_.get("warning") is not None:
            return Xs  # fit failed – return untransformed

        if "JDOT" in type(self.model).__name__:
            return self._transform_jdot(Xs)

        transp_Xs = self.model.transform(
            Xs, sample_domain=np.ones(len(Xs)), allow_source=True, **kwargs
        )
        return transp_Xs

    # -- fit_transform --------------------------------------------------------
    def fit_transform(self, Xs, Xt=None, ys=None, yt=None, Ys=None, Yt=None, **kwargs):
        self.fit(Xs, Xt, ys=ys, yt=yt, Ys=Ys, Yt=Yt, **kwargs)
        return self.transform(Xs, Xt=Xt)

    # -- internal: JDOT coupling-based transform ------------------------------
    def _transform_jdot(self, Xs):
        nx = get_backend(Xs)
        plan = self.model.sol_.plan
        transp = plan / nx.sum(plan, axis=1)[:, None]
        self.coupling_ = transp
        transp = nx.nan_to_num(transp, nan=0, posinf=0, neginf=0)

        if Xs is self.xs_:
            return nx.dot(transp, self.xt_)

        # out-of-sample mapping via nearest-neighbour interpolation
        indices = nx.arange(Xs.shape[0])
        batch_size = 256
        batch_ind = [
            indices[i : i + batch_size]
            for i in range(0, len(indices), batch_size)
        ]
        transp_Xs = []
        for bi in batch_ind:
            D0 = dist(Xs[bi], self.xs_)
            idx = nx.argmin(D0, axis=1)
            transp_Xs_ = nx.dot(transp, self.xt_)
            transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]
            transp_Xs.append(transp_Xs_)
        return nx.concatenate(transp_Xs, axis=0)


class baseSkadaPredictor(baseSkada):
    """Base for skada methods that only support ``predict`` (no feature
    mapping).  ``transform`` raises ``NotImplementedError``.
    """

    def transform(self, Xs, Xt=None, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is a predictor-only DA method and does "
            "not support feature-space transport via transform(). "
            "Use predict() instead."
        )

    def fit_transform(self, Xs, Xt=None, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not support transform – use fit() + predict()."
        )


# ===================================================================
# Concrete wrapper classes – mapping / alignment adapters
# ===================================================================

class JDOT(baseSkada):
    """Joint Distribution OT regressor wrapper."""
    _adapter = JDOTRegressor


class JDOTC(baseSkada):
    """Joint Distribution OT classifier wrapper."""
    _adapter = JDOTClassifier


class OTMapping(baseSkada):
    """OT mapping transport wrapper (uses OTMappingAdapter)."""
    _adapter = OTMappingAdapter


class EntropicOT(baseSkada):
    """Entropic OT mapping adapter wrapper."""
    _adapter = EntropicOTMappingAdapter


class ClassOT(baseSkada):
    """Class-regularized OT mapping adapter wrapper."""
    _adapter = ClassRegularizerOTMappingAdapter


class LinearOT(baseSkada):
    """Linear OT mapping adapter wrapper."""
    _adapter = LinearOTMappingAdapter


class CORALDA(baseSkada):
    """CORAL domain adaptation wrapper."""
    _adapter = CORALAdapter


class TCA(baseSkada):
    """Transfer Component Analysis wrapper."""
    _adapter = TransferComponentAnalysisAdapter


class SubspaceAlignmentDA(baseSkada):
    """Subspace Alignment domain adaptation wrapper."""
    _adapter = SubspaceAlignmentAdapter


class TransferJointMatchingDA(baseSkada):
    """Transfer Joint Matching domain adaptation wrapper."""
    _adapter = TransferJointMatchingAdapter


class TransferSubspaceLearningDA(baseSkada):
    """Transfer Subspace Learning domain adaptation wrapper."""
    _adapter = TransferSubspaceLearningAdapter


class MMDLSConSDA(baseSkada):
    """MMD Location-Scale mapping wrapper."""
    _adapter = MMDLSConSMappingAdapter


class MultiLinearMongeDA(baseSkada):
    """Multi-domain Linear Monge Alignment wrapper."""
    _adapter = MultiLinearMongeAlignmentAdapter


# ===================================================================
# Concrete wrapper classes – predictor-only methods
# ===================================================================

class DASVMClassifierDA(baseSkadaPredictor):
    """DASVM classifier wrapper (predict only, no transform).

    Uses the ``DASVMClassifier`` pipeline estimator from skada.
    Requires ``Ys`` / ``Yt`` labels for fitting.
    """
    _adapter = _DASVMClassifier

    def __init__(self, **kwargs):
        if self._adapter is None:
            raise TypeError(
                "DASVMClassifier is not available in this skada installation"
            )
        # DASVMClassifier is a function that creates a pipeline;
        # call it to get the estimator
        self.model = self._adapter(**kwargs)
        self.kwargs = kwargs
        self.log_ = {"warning": None}


class OTLabelPropDA(baseSkadaPredictor):
    """OT label propagation wrapper (predict only, no transform).

    Uses the ``OTLabelPropAdapter`` from skada.
    """
    _adapter = _OTLabelPropAdapter


class JCPOTLabelPropDA(baseSkadaPredictor):
    """JCPOT label propagation wrapper (predict only, no transform)."""
    _adapter = _JCPOTLabelPropAdapter


# ===================================================================
# Populate METHODS dict
# ===================================================================
_ALL_WRAPPERS = {
    # --- original methods ---
    "jdot": JDOT,
    "jdotc": JDOTC,
    "otmapping": OTMapping,
    "entropicOT": EntropicOT,
    "classOT": ClassOT,
    "linearOT": LinearOT,
    "coral": CORALDA,
    "TCA": TCA,
    # --- new mapping / alignment ---
    "subspaceAlignment": SubspaceAlignmentDA,
    "transferJointMatching": TransferJointMatchingDA,
    "transferSubspaceLearning": TransferSubspaceLearningDA,
    "mmdLSConS": MMDLSConSDA,
    "multiLinearMonge": MultiLinearMongeDA,
    # --- predictor-only ---
    "dasvm": DASVMClassifierDA,
    "otLabelProp": OTLabelPropDA,
    "jcpotLabelProp": JCPOTLabelPropDA,
}

# Only register wrappers whose underlying skada class is actually available
for _name, _cls in _ALL_WRAPPERS.items():
    if _cls._adapter is not None:
        METHODS[_name] = _cls

# Also expose the list of transform-capable methods for flexible transporter search
TRANSFORM_CAPABLE_METHODS = {
    k: v for k, v in METHODS.items()
    if not issubclass(v, baseSkadaPredictor)
}


# ===================================================================
# Skada metrics as tune()-compatible error functions
# ===================================================================

def _build_scorer_error_func(scorer_class, **scorer_kwargs):
    """Create a ``(Xs, Xt, Ys, Yt) -> float`` error function from a skada
    scorer.  The scorer is applied by fitting a simple RF classifier on the
    transported source and evaluating on target.
    """
    def _error_func(Xs, Xt, Ys, Yt):
        try:
            scorer = scorer_class(**scorer_kwargs)
            # Build a simple classifier on transported source
            clf = _RFC(n_estimators=50, random_state=0)
            if Ys is None or Yt is None:
                ys = np.zeros(Xs.shape[0])
                yt = np.zeros(Xt.shape[0])
            else:
                ys, yt = Ys, Yt
            X = np.concatenate((Xs, Xt), axis=0)
            Y = np.concatenate((ys, yt), axis=0)
            domain = np.concatenate(
                (np.ones(Xs.shape[0]), -np.ones(Xt.shape[0])), axis=0
            )
            clf.fit(Xs, ys)
            # skada scorers expect (estimator, X, y, sample_domain=...)
            score = scorer(clf, X, Y, sample_domain=domain)
            # Scorers return higher-is-better by convention; negate for minimisation
            return -score
        except Exception as e:
            logger.debug("skada metric scorer failed: %s", e)
            return 9e5
    return _error_func


if _HAS_SKADA_METRICS:
    prediction_entropy_error = _build_scorer_error_func(PredictionEntropyScorer)
    soft_neighborhood_density_error = _build_scorer_error_func(SoftNeighborhoodDensity)
    circular_validation_error = _build_scorer_error_func(CircularValidation)
else:
    def _missing_metric(*args, **kwargs):
        raise ImportError(
            "skada.metrics is not available – install a recent version of skada"
        )
    prediction_entropy_error = _missing_metric
    soft_neighborhood_density_error = _missing_metric
    circular_validation_error = _missing_metric
