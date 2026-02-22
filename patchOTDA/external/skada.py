"""Wrappers around scikit-adaptation (skada) methods for the patchOTDA API.

All classes follow the ``fit(Xs, Xt, Ys, Yt)`` / ``transform(Xs)`` pattern.
Requires the ``skada`` package to be installed.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from skada import JDOTRegressor, JDOTClassifier, CORAL, CORALAdapter, TransferComponentAnalysisAdapter
    from skada import (
        OTMapping as _OTMapping,
        EntropicOTMappingAdapter,
        ClassRegularizerOTMappingAdapter,
        LinearOTMappingAdapter,
        make_da_pipeline,
    )
    from skada import source_target_split
except ImportError:
    logger.warning("skada not installed, external.skada wrappers will not work")
    raise

import numpy as np
from ot.backend import get_backend
from ot import dist

METHODS = {
    "jdot": JDOTRegressor,
    "jdotc": JDOTClassifier,
    "otmapping": _OTMapping,
    "entropicOT": EntropicOTMappingAdapter,
    "classOT": ClassRegularizerOTMappingAdapter,
    "linearOT": LinearOTMappingAdapter,
    "coral": CORALAdapter,
    "TCA": TransferComponentAnalysisAdapter
}


class baseSkada:
    """Base class wrapping skada adapters into the patchOTDA fit/transform API."""
    def __init__(self, **kwargs):
        self.model =  self.model(**kwargs)
        self.kwargs = kwargs

    def fit(self, Xs, Xt, Ys, Yt, **kwargs):
        #concat the source and target data
        X = np.concatenate((Xs, Xt), axis=0)
        Y = np.concatenate((Ys, Yt), axis=0)
        #make a sample domain label
        domain = np.concatenate((np.ones(Xs.shape[0]), np.ones(Xt.shape[0])*-1), axis=0)

        self.model.fit(X, Y, sample_domain=domain, **kwargs)
        self.xt_ = Xt
        self.xs_ = Xs

    def predict(self, Xs, **kwargs):
        return self.model.predict(Xs, **kwargs)

    def transform(self, Xs, **kwargs):

        if "JDOT" in self.model.__class__.__name__:
            #we can use the OT mapping to transform the data
            nx = get_backend(Xs)
            plan = self.model.sol_.plan
            
            transp = plan / nx.sum(plan, axis=1)[:, None]
            self.coupling_ = transp
            # set nans to 0
            transp = nx.nan_to_num(transp, nan=0, posinf=0, neginf=0)

            # compute transported samples
            if Xs is self.xs_:
                transp_Xs = nx.dot(transp, self.xt_)
            else:
                                # perform out of sample mapping
                indices = nx.arange(Xs.shape[0])
                batch_size = 256
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []
                for bi in batch_ind:
                    # get the nearest neighbor in the source domain
                    D0 = dist(Xs[bi], self.xs_)
                    idx = nx.argmin(D0, axis=1)

                    # transport the source samples
                    transp = nx.nan_to_num(transp, nan=0, posinf=0, neginf=0)
                    transp_Xs_ = nx.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = nx.concatenate(transp_Xs, axis=0)
        else:
            transp_Xs = self.model.transform(Xs, sample_domain=np.ones(len(Xs)), allow_source=True, **kwargs)


        return transp_Xs
    
#make a class for each method
class JDOT(baseSkada):
    """Joint Distribution OT regressor wrapper."""
    model = JDOTRegressor

class JDOTC(baseSkada):
    """Joint Distribution OT classifier wrapper."""
    model = JDOTClassifier

class OTMapping(baseSkada):
    """OT mapping transport wrapper."""
    model = _OTMapping

class EntropicOT(baseSkada):
    """Entropic OT mapping adapter wrapper."""
    model = EntropicOTMappingAdapter

class ClassOT(baseSkada):
    """Class-regularized OT mapping adapter wrapper."""
    model = ClassRegularizerOTMappingAdapter

class LinearOT(baseSkada):
    """Linear OT mapping adapter wrapper."""
    model = LinearOTMappingAdapter

class CORALDA(baseSkada):
    """CORAL domain adaptation wrapper."""
    model = CORALAdapter

class TCA(baseSkada):
    model = TransferComponentAnalysisAdapter
