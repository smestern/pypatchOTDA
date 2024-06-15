from skada import JDOTRegressor, JDOTClassifier
from skada import (
    OTMapping,
    EntropicOTMappingAdapter,
    ClassRegularizerOTMappingAdapter,
    LinearOTMappingAdapter,
    make_da_pipeline,
)
import numpy as np
from ot.backend import get_backend
METHODS = {
    "jdot": JDOTRegressor,
    "jdotc": JDOTClassifier,
    "otmapping": OTMapping,
    "entropicOT": EntropicOTMappingAdapter,
    "classOT": ClassRegularizerOTMappingAdapter,
    "linearOT": LinearOTMappingAdapter,

}


class baseSkada:
    def __init__(self, **kwargs):
        self.model =  self.model(**kwargs)
        self.kwargs = kwargs

    def fit(self, Xs, Xt, Ys, Yt, **kwargs):
        #concat the source and target data
        X = np.concatenate((Xs, Xt), axis=0)
        Y = np.concatenate((Ys, Yt), axis=0)
        #make a sample domain label
        domain = np.concatenate((np.ones(Xs.shape[0]), np.ones(Xt.shape[0])*-1), axis=0)
        self.model.fit(X, Y, domain, **kwargs)
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
                    transp = self.coupling_ / nx.sum(self.coupling_, axis=1)[:, None]
                    transp = nx.nan_to_num(transp, nan=0, posinf=0, neginf=0)
                    transp_Xs_ = nx.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = nx.concatenate(transp_Xs, axis=0)
            


        return transp_Xs
    
#make a class for each method
class JDOT(baseSkada):
    model = JDOTRegressor

class JDOTC(baseSkada):
    model = JDOTClassifier

class OTMapping(baseSkada):
    model = OTMapping

class EntropicOT(baseSkada):
    model = EntropicOTMappingAdapter

class ClassOT(baseSkada):
    model = ClassRegularizerOTMappingAdapter

class LinearOT(baseSkada):
    model = LinearOTMappingAdapter


