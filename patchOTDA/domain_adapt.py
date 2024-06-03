import numpy as np

from functools import partial
import copy
import warnings
import joblib
import multiprocessing.pool
import functools
import scipy.spatial as spatial
import logging
from inspect import signature


import nevergrad as ng
import sklearn
from sklearn.base import BaseEstimator
from ot.da import BaseTransport, distribution_estimation_uniform, check_params
import ot
from ot.utils import dist, cost_normalization

from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#optional imports
try:
    import torch
except:
    logging.warning("torch not installed, unbalancedFUGWTransporter will not work")

try:
    from unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn, log_ugw_sinkhorn
    from unbalancedgw._vanilla_utils import ugw_cost
except:
    logging.warning("unbalancedgw not installed, unbalancedFUGWTransporter will not work")
    
MAX_ITER = int(1e3)
MAX_INNER_ITER = int(1e5)
TIMEOUT = 60*2

_reg1_forward = ng.p.Log(lower=1e-9, upper=10)
_reg1_backward = ng.p.Log(lower=1e-9, upper=10)
_reg2_forward = ng.p.Log(lower=1e-9, upper=10)
_reg2_backward = ng.p.Log(lower=1e-9, upper=10)
_limit_max = ng.p.Log(lower=1, upper=10)
_norm = ng.p.Choice(['max', 'median', None])
_metric = ng.p.Choice(['sqeuclidean', 'seuclidean', 'euclidean', 'cosine', 'cityblock', 'minkowski', 'correlation'])
_semi_super = ng.p.Choice([False, False])
DEFAULT_OPTIMIZABLE_KWARGS = {'reg': ng.p.Log(lower=1e-10, upper=1), 
'metric':ng.p.Choice(['sqeuclidean','euclidean']) , 

'norm':ng.p.Choice(['max', 'median', None]), 
'similarity_param': ng.p.Scalar(lower=2, upper=200 )}


class PatchClampOTDA(BaseEstimator, BaseTransport):
    def __init__(self, transporter=None, flexible_transporter=False, **kwargs):
        """Domain Adaptation with OT with a specific focus on patch clamp based data. Here we use the OT-DA framework.
        This wrapper for pyOT based domain adaptation. Fit and transform functions are implemented to transform one piece of data to another.
        Includes tune function for tuning the parameters to the dataset provided.

        Args:
            transporter (str, ot.da.Transport, optional): Transporter used to fit and transform the data. Must be a pyOT based transporter. Defaults to ot.da.EMDLaplaceTransport.
            flexible_transporter (bool, optional): Whether the transporter can be changed (to a different method) while tuning. Defaults to False.
            kwargs (dict, optional): kwargs to be passed to the transporter. Defaults to {}.
        """
        if transporter is None:
            transporter = ot.da.EMDLaplaceTransport

        if isinstance(transporter, str):
            #eval is dangerous, maybe remove?
            self.inittransporter = eval('ot.da.'+transporter)
        else:
            self.inittransporter = transporter
        
        #super().__init__(**kwargs)
        #dont super init here, instantiate transporter first
        self.transporter = self.inittransporter(**getValidKwargs(transporter, kwargs))
        self._kwargs = kwargs
        self.flexible_transporter = flexible_transporter
        

    def fit(self, Xs, Xt, Ys=None, Yt=None):
        self.transporter.fit(Xs=Xs, Xt=Xt, ys=Ys, yt=Yt)
        return self
    
    def transform(self, Xs, Xt, Ys=None, Yt=None):
        return self.transporter.transform(Xs=Xs, Xt=Xt, ys=Ys, yt=Yt)

    def fit_transform(self, Xs, Xt, Ys=None, Yt=None):
        self.transporter.fit(Xs=Xs, Xt=Xt, ys=Ys, yt=Yt)
        return self.transporter.transform(Xs=Xs, Xt=Xt, ys=Ys, yt=Yt)

    def fit_tune_transform(self, Xs, Xt, Ys=None, Yt=None, n_iter=10, n_jobs=-1, verbose=False):
        self.tune(Xs, Xt, Ys, Yt, n_iter, n_jobs, verbose)
        self.transporter.fit(Xs=Xs, Xt=Xt, ys=Ys, yt=Yt)
        return self.transporter.transform(Xs=Xs, Xt=Xt, ys=Ys, yt=Yt)
    
    def tune(self, Xs, Xt, Ys=None, Yt=None, n_iter=20, n_jobs=-1, method='bidirectional', error_func=None, verbose=False):
        """ Tune the parameters of the OTDA based transporter to the datasets provided.
        Tunes the reg parameters and the norm of the data sets. Currently supports using an error function base on
        the mse between randomly skewed data and Xt.
        Supports paralellism via n_jobs. The total number of points queried is (n_iter x n_jobs)
        Uses nevergrad as optimizer backend. TODO// allow user to select optimizer

        Args:
            Xs (numpy array): _description_
            Xt (numpy array): _description_
            Ys (_type_, optional): _description_. Defaults to None.
            Yt (_type_, optional): _description_. Defaults to None.
            n_iter (int, optional): _description_. Defaults to 10.
            n_jobs (int, optional): _description_. Defaults to -1.
            method (str, optional): _description_. Defaults to 'bidirectional'.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if method == 'bidirectional':
            #if the user wants to use the bidirectional method, we do not need to update the dict. 
            #we just need to update the error function, enforce it is None, and set the method to bidirectional
            if error_func is not None:
                raise ValueError('Cannot use error_func with bidirectional method')
            #now just run the bidirectional tune
            self.param_dict = self.init_param_dict(Xs, Xt, method, error_func)
            return self._tune(Xt, Xs, Ys, Yt, n_iter, n_jobs, method=method, verbose=verbose)
        elif method == 'unidirectional':
            
            #now just run the unidirectional tune
            self.param_dict = self.init_param_dict(Xs, Xt, method, error_func)
            return self._tune(Xs, Xt, Ys, Yt, n_iter, n_jobs, error_func, method, verbose)
        

        return self
    
    def _tune(self, Xs, Xt, Ys=None, Yt=None, n_iter=20, n_jobs=-1, error_func=None, method='bidirectional', verbose=False):
        """ Tune the parameters of the OTDA based transporter to the datasets provided.
        Tunes the reg parameters and the norm of the data sets. Currently supports using an error function base on
        the mse between randomly skewed data and Xt.
        Supports paralellism via n_jobs. The total number of points queried is (n_iter x n_jobs)
        Uses nevergrad as optimizer backend.

        Args:
            Xs (numpy array): _description_
            Xt (numpy array): _description_
            Ys (_type_, optional): _description_. Defaults to None.
            Yt (_type_, optional): _description_. Defaults to None.
            n_iter (int, optional): _description_. Defaults to 10.
            n_jobs (int, optional): _description_. Defaults to -1.


            method (str, optional): _description_. Defaults to 'bidirectional'.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        #if the user does not specify just use the default values
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        #set the tune function depending on the method
        if method == 'bidirectional':
            tune_func = _inner_tune_back_and_forth
            direction = 'forward'
        elif method == 'unidirectional':
            tune_func = _tune_transporter
            direction = None

        #init a nevergrad optimization
        self.opt = ng.optimizers.Portfolio(self.param_dict, budget=n_iter*n_jobs, num_workers=n_jobs,)
        with warnings.catch_warnings():#Catch the sinkhorn warnings
            warnings.simplefilter("ignore")
            for i in np.arange(n_iter):

                #get the current params
                param_list = []
                for n in np.arange(n_jobs):
                    param_list.append(self.opt.ask())
                
                
                #get the current score
                if self.flexible_transporter:
                    #get the current transporter
                    
                    score = joblib.Parallel(n_jobs=n_jobs,  prefer="threads", require='sharedmem', verbose=verbose)(joblib.delayed(tune_func)(Xs, Xt, Ys, Yt, error_func=error_func, opt_kwargs=p.value) for p in param_list)
                else:
                    transporter = self.inittransporter
                    score = joblib.Parallel(n_jobs=n_jobs,  prefer="threads", require='sharedmem', mmap_mode=None, verbose=verbose)(joblib.delayed(tune_func)(Xs, Xt, Ys, Yt, error_func=error_func, transporter=transporter, opt_kwargs=p.value) for p in param_list)
                #update the nevergrad params
                for p, e in zip(param_list, score):
                    self.opt.tell(p, e)
                #self.opt.tell(params, score)
                if verbose:
                    print('iteration: ', i, 'score: ', np.amin(score))
        #get the best transporter
        print("best kwargs:")
        best_args = self.opt.recommend().value
        print(best_args)
        transporter, best_kwargs = self._to_kwarg_dict(self.inittransporter, kwargs=copy.copy(best_args), direction=direction)
        self.transporter = transporter(**best_kwargs,  log=True)
        self.best_ = best_kwargs
        return self

    
    def _to_kwarg_dict(self, transporter, kwargs, error_func=None, verbose=False, direction='forward'):
        if direction is not None:
            if direction.upper() == 'BACKWARD':
                    #grab the kwargs with backward in the key
                    kwargs = {k[:k.find('_backward')]:v for k,v in kwargs.items() if '_backward' in k}
            elif direction.upper() == 'FORWARD':
                    kwargs = {k[:k.find('_forward')]:v for k,v in kwargs.items() if '_forward' in k}
            transporter = kwargs.pop('transporter')
        else:
            transporter = kwargs.pop('transporter')
        return transporter, getValidKwargs(transporter, kwargs)


    def init_param_dict(self, Xt, Xs, method, error_func):
        """Initialize the param dict for the nevergrad optimizer

        Args:
            Xt (_type_): _description_
            Xs (_type_): _description_
            method (_type_): _description_
            error_func (_type_): _description_

        Returns:
            dict: _description_
        """        
        #DEFAULTS
        methods = [ot.da.UnbalancedSinkhornTransport, ot.da.SinkhornTransport, ot.da.EMDTransport, ot.da.EMDLaplaceTransport]
        
        #update the defaults similiarity params to max of the two datasets
        DEFAULT_OPTIMIZABLE_KWARGS.update({'similarity_param':ng.p.Scalar(lower=2, upper=np.amin([Xt.shape[0], Xs.shape[0]]))})

        #for now support  reg, norm,
        if method == 'bidirectional':
            #if the user wants to use the bidirectional method,
            # #here we want to clone each option to add a backward and forward option
            options = []
            for x in methods:
                method_kwargs = getOptimizableKwargs(x)
                #add the forward and backward options
                #essentially just clone each key and affix forward and backward
                method_kwargs_forward = {k+'_forward':copy.deepcopy(v) for k,v in method_kwargs.items()}
                method_kwargs_backward = {k+'_backward':v for k,v in method_kwargs.items()}
                full_kwargs = {**method_kwargs_forward, **method_kwargs_backward}
                options.append(ng.p.Dict(**full_kwargs))
            #if the user does not want to use the specific transporter, we can add the flexibility to use a different transporter
            if self.flexible_transporter:
                #update the paramdict to include the list of transporter
                _param_dict = ng.p.Choice(options)
            else:
                #we want to figure out what params go with what transporter 
                #else just the default param 
                raise NotImplementedError("Bidirectional tuning not implemented yet for a specific transporter")
                
        elif method == 'unidirectional':
            options = [ng.p.Dict(**getOptimizableKwargs(x)) for x in methods]
            #if the user wants unidirectional tune
            if self.flexible_transporter:
                 _param_dict = ng.p.Choice(options)
            else:
                #we want to figure out what params go with what transporter
                _param_dict = ng.p.Dict(**getOptimizableKwargs(self.inittransporter))
            #now just run the unidirectional tune
            
        return _param_dict
    

#UTIL FUNCTIONS

##From https://stackoverflow.com/a/35139284
def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            try:
                res = async_result.get(max_timeout)
            except multiprocessing.TimeoutError:
                res = 999999
                pool.terminate()
                pool.close()
            pool.terminate()
            pool.close()
            return res
        return func_wrapper
    return timeout_decorator

def getValidKwargs(func, argsDict):
    sig =  signature(func)
    kwargs_names = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
    new_args = {}
    for key, value in argsDict.items():
        if key in kwargs_names:
            new_args[key] = copy.deepcopy(value)
        else:
            pass
    return new_args

def getOptimizableKwargs(func, optimizableKwargs=DEFAULT_OPTIMIZABLE_KWARGS):
    sig =  signature(func)
    kwargs_names = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
    new_args = {}
    list_opt_kwargs = list(optimizableKwargs.keys())
    for key in kwargs_names:
        #check if the key is in the optimizable kwargs
        key_match = np.where([x.upper() in key.upper() for x in  list_opt_kwargs])[0]
        if len(key_match) > 0:
            new_args[key] = copy.deepcopy(optimizableKwargs[list_opt_kwargs[key_match[0]]])
        else:
            pass
    new_args['transporter'] = copy.deepcopy(func)
    return new_args

        


@timeout(TIMEOUT) #timeout after 5 minutes
def _inner_tune_back_and_forth(Xs, Xt, Ys, Yt, transporter=ot.da.SinkhornTransport, sample_size=None,  
error_func=None, verbose=False, opt_kwargs={},):
    """Inner function for tuning the transporter. This method attempts to find the optimization parameters for the transporter.
    First it tries to find the best parameters for the forward direction, transporting a random sample of the target to the source.
    then tries to find the best parameters for the backward direction, transporting that random sample back to the target.
    Unfortunately, this method is very slow, it has to optimize two different parameters for each direction.
    Args:
        transporter (_type_): _description_
        Xs (_type_): _description_
        Xt (_type_): _description_
        Ys (_type_): _description_
        Yt (_type_): _description_
        sample_size (_type_, optional): _description_. Defaults to None.
        reg1_forward (_type_, optional): _description_. Defaults to 1e-1.
        reg2_forward (_type_, optional): _description_. Defaults to 1e-1.
        reg1_backward (_type_, optional): _description_. Defaults to 1e-1.
        reg2_backward (_type_, optional): _description_. Defaults to 1e-1.
        norm (str, optional): _description_. Defaults to 'median'.
        metric (str, optional): _description_. Defaults to 'sqeuclidean'.
        limit_max (int, optional): _description_. Defaults to 10.
        error_func (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    #pull out the forward and backward kwargs, the forward kwargs will have _forward affixed to the end of the key
    #the backward kwargs will have _backward affixed to the end of the key
    #we will use these to update the transporter kwargs
    forward_kwargs = {k[:k.find('_forward')]:v for k,v in opt_kwargs.items() if '_forward' in k}
    backward_kwargs = {k[:k.find('_backward')]:v for k,v in opt_kwargs.items() if '_backward' in k}
    
    if error_func is None:
        error_func = mean_squared_error
    #print("Tuning the transporter")
    if sample_size is None:
        #make it the same ratio of Xs and Xt
        sample_size = np.clip(Xs.shape[0]/Xt.shape[0], 0.1, 0.5) #at least 1 sample, at most half of the smaller one

    #we are gonna mess with the transporter, so make a copy
    Xt = Xt.copy()
    Xs = Xs.copy()

    rand_idx = np.random.choice(Xt.shape[0], int(sample_size*Xt.shape[0]), replace=False)
    Xt_sample = Xt[rand_idx, :]
    if Yt is not None:
        Yt_sample = Yt[rand_idx]
    else:
        Yt_sample = None
    Xt_base = np.delete(Xt, rand_idx, axis=0)

    #reinit the transporter with the reg_kwargs
    transporter_one = transporter(**getValidKwargs(transporter, forward_kwargs), log=True)
    #fit the transporter but the other direction
    #transporter_one.fit(Xs=Xt_sample, Xt=Xs)
    #transform the data
    Xt_transport = transporter_one.fit_transform(Xs=Xt_sample, Xt=Xs, ys=Yt_sample, yt=Ys)

    #reinit the transporter with the reg_kwargs
    transporter_two = transporter(**getValidKwargs(transporter, backward_kwargs), log=True)
    #fit the transporter
    #transporter_one.fit(Xs=Xt_transport, Xt=Xt_base)
    #transform the data
    Xt_reconstructed = transporter_two.fit_transform(Xs=Xt_transport, Xt=Xt,  ys=Yt_sample, yt=Yt)
    
    error = error_func(Xt_sample, Xt_reconstructed)
    #Check for erros in fitting
    if np.all(np.isnan(Xt_transport)) or np.all(np.isnan(Xt_reconstructed)):
        error = 9*1e5
    elif np.all(Xt_reconstructed==0):
        error = 9*1e5
    elif np.all(Xt_transport==0):
        error = 9*1e5

    #also try one final transport of the whole data
    transporter_three = transporter(**getValidKwargs(transporter, backward_kwargs), log=True)
    #fit the transporter
    transporter_three.fit(Xs=Xt, Xt=Xs)
    
    #check the log
    if 'warning' in transporter_one.log_:
        if (transporter_one.log_['warning'] is not None) or (transporter_two.log_['warning'] is not None) or (transporter_three.log_['warning'] is not None):
            error = 9*1e5

    return error

@timeout(300)
def _tune_transporter(Xs, Xt, Ys, Yt, transporter=ot.da.SinkhornTransport,  error_func=None, opt_kwargs={},):
    """_summary_

    Args:
        transporter (_type_): _description_
        Xs (_type_): _description_
        Xt (_type_): _description_
        Ys (_type_): _description_
        Yt (_type_): _description_
        sample_size (_type_, optional): _description_. Defaults to None.
        reg1_backward (_type_, optional): _description_. Defaults to 1e-1.
        reg2_backward (_type_, optional): _description_. Defaults to 1e-1.
        norm (str, optional): _description_. Defaults to 'median'.
        metric (str, optional): _description_. Defaults to 'sqeuclidean'.
        limit_max (int, optional): _description_. Defaults to 10.
        error_func (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
    """
    #print("Tuning the transporter")
    if 'transporter' in list(opt_kwargs.keys()):
        transporter = opt_kwargs.pop('transporter')

    if error_func is None:
        error_func = ugw_dist
    #print("Tuning the transporter")
    #simply transport
    transporter_ = transporter(**getValidKwargs(transporter, opt_kwargs), log=True)
    #fit the transporter
    transporter_.fit(Xs=Xs, Xt=Xt)
    #transform the data
    Xs_transport = transporter_.fit_transform(Xs=Xs, Xt=Xt)
    #check the error
    error = error_func(Xs_transport, Xt, Ys, Yt)
    #Check for erros in fitting

    if np.all(np.isnan(Xs_transport)):
        error = 9e5
    elif np.all(Xs_transport==0):
        error = 9e5
    #also try one final transport of the whole data
    transporter_.fit(Xs=Xs, Xt=Xt)
    #check the log
    if 'warning' in transporter_.log_:
        #if there is a warning it is an integration error, despite low recorded error the result is funky so just punish the error term
        if transporter_.log_['warning'] is not None:
            error = 9*1e5
    return error 


##ERROR FUNCTIONS

def normalized_mse(x,y):
    #normalize based on column mean https://www.mathworks.com/help/ident/ref/goodnessoffit.html#d123e47621
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    mse = ((x-y)**2)/((x - x_mean)**2)
    return np.nanmean(mse)

def rowwise_mse(x, y):
    return np.mean(np.square(x-y))

def mape(x, y):
    return np.abs(np.nanmean((np.abs(x-y))/(x)))

def gw_dist(Xs, Xt, Ys, Yt):
    #compute the GW distance between two sets of samples
    #compute the cost matrix
    C1 = ot.dist(Xs, Xs)
    C2 = ot.dist(Xt, Xt)

    C1 /= C1.max()
    C2 /= C2.max()
    p = ot.unif(Xs.shape[0])
    q = ot.unif(Xt.shape[0])
    gw, log = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, loss_fun='kl_loss', log=True, verbose=False)
        
    return log['gw_dist']

def ugw_dist(Xs, Xt, Ys, Yt):
    
    #compute the GW distance between two sets of samples
    #compute the cost matrix
    C1 = torch.tensor(spatial.distance.cdist(Xs, Xs))
    C2 = torch.tensor(spatial.distance.cdist(Xt, Xt))
    Xs = torch.tensor(Xs)
    Xt = torch.tensor(Xt)

    C1 /= C1.max()
    C2 /= C2.max()
    p = torch.tensor(ot.unif(Xs.shape[0]))
    q = torch.tensor(ot.unif(Xt.shape[0]))
    eps = 1.0
    rho = 0.1
    rho2 = 0.1
    #nan to num the cost matrices
    C1 = torch.nan_to_num(C1, nan=0, posinf=0, neginf=0)
    C2 = torch.nan_to_num(C2, nan=0, posinf=0, neginf=0)
    #compute the gromov wasserstein distance
    pi, gamma = log_ugw_sinkhorn(p, C1, q, C2, init=None, eps=eps,
                             rho=rho, rho2=rho2,
                             nits_plan=1000, tol_plan=1e-5,
                             nits_sinkhorn=1000, tol_sinkhorn=1e-5,
                             two_outputs=True)
    cost = ugw_cost(pi, gamma, p, C1, q, C2, eps=eps, rho=rho, rho2=rho2)
        
    return float(cost.numpy())
    
def compute_structure_index(Xs, Xt, Ys, Yt):
    #TODO: add a check for the number of samples in each set
    raise NotImplementedError
    #compute the structure index between two sets of samples
    cost = structure_index.compute_structure_index(Xs.T, Xt.T)
        
    return cost

def rf_dist(Xs, Xt, Ys, Yt):
    #compute the random forest distance between two sets of samples
    #compute the cost matrix
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    rf.fit(Xs, Ys)
    Yt_hat = rf.predict(Xt)
    return np.mean((Yt_hat-Yt)**2)

def rf_clf_dist(Xs, Xt, Ys, Yt):
    #compute the random forest classifier distance between two sets of samples
    #compute the cost matrix
    #mask the -1 labels
    mask = Ys!=-1
    mask2 = Yt!=-1

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(Xt[mask2], Yt[mask2])
    Ys_hat = rf.predict(Xs[mask])
    return 1 - balanced_accuracy_score(Ys[mask], Ys_hat)

class metrics(object):
    #container for the distance metrics
    #list the metrics as class methods
    #unsupervised
    normalized_mse = normalized_mse
    rowwise_mse = rowwise_mse
    mape = mape
    GW_Dist = gw_dist
    

    #semi-supervised/supervised
    RF_Dist = rf_dist
    RF_Clf_Dist = rf_clf_dist

class dummyScaler(sklearn.preprocessing.StandardScaler):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X
    def fit_transform(self, X, y=None):
        return X
    def inverse_transform(self, X, y=None):
        return X
    

#make a unbalancedFUGW transporter
#TODO: this should be updated to meet the conventions of the other transporters
#eg background agnostic, etc
#TODO: add a check for the number of samples in each set
class unbalancedFUGWTransporter(BaseTransport):
    def __init__(self, reg_e=0.1, reg_m1=0.1, reg_m2=0.1, 
                 max_iter=10, tol=1e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=10) -> None:
        super().__init__()
        self.log_ = {}
        self.log_['warning'] = None
        self.log_['error'] = None
        self.log_['error_type'] = None
        self.log_['error_message'] = None
        self.reg_e = reg_e
        self.reg_m1 = reg_m1
        self.reg_m2 = reg_m2
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        if norm is None or norm == "median":
           # self.norm = np.median
            self.norm = norm
        else:
            print("norm must be None or 'median for unbalancedFUGWTransporter'")
            print("setting norm to None")
            self.norm = None

        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`
        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.
            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):
            #here we are gonna force torch
            Xs = torch.tensor(Xs)
            Xt = torch.tensor(Xt)

            super(unbalancedFUGWTransporter, self).fit(Xs, ys, Xt, yt)
            
            #we actually need to compute the cost matrix for samples in Xs and Xt
            self.cost_s = cost_normalization(dist(Xs, Xs, metric=self.metric), norm=self.norm)
            self.cost_t = cost_normalization(dist(Xt, Xt, metric=self.metric), norm=self.norm)
            try:
                returned_ = (log_ugw_sinkhorn(
                    a=self.mu_s, dx=self.cost_s, b=self.mu_t, dy=self.cost_t,
                    eps=self.reg_e, rho=self.reg_m1, rho2=self.reg_m2, 
                    nits_sinkhorn=self.max_iter, tol_sinkhorn=self.tol), dict())
            except Exception as e:
                print(e)
                print("error in log_ugw_sinkhorn")
                print("setting coupling matrix to ones")
                self.coupling_ = torch.ones(Xs.shape[0], Xt.shape[0], dtype=torch.float64)
                returned_ = (self.coupling_, dict())
            

            # deal with the value of log
            if self.log:
                self.coupling_, self.log_ = returned_
            else:
                self.coupling_ = returned_
                self.log_ = dict()

        return self
    
    def transform(self, Xs, ys=None, Xt=None, yt=None):
        #check the necessary inputs parameters are here
        if check_params(Xs=Xs):
            #here we are gonna force torch
            Xs = torch.tensor(Xs)
            Xt = torch.tensor(Xt)
            #pass to super
            return super(unbalancedFUGWTransporter, self).transform(Xs, ys, Xt, yt).numpy()




# def tune_grad_descent(self, Xs, Xt, Ys=None, Yt=None, n_iter=500, verbose=True):
#         #for certain methods, we can use gradient descent to find the best params
        
#         #put our data into torch
#         Xs = torch.from_numpy(Xs).float()
#         Xt = torch.from_numpy(Xt).float()

#         dist_x1 = torch.ones(Xt.shape[0]).float() / Xt.shape[0]
#         dist_x2 = torch.ones(Xs.shape[0]).float() / Xs.shape[0]

#         dist_x1.requires_grad = True
#         dist_x2.requires_grad = True

#         #grab a random sample
#         idx = np.random.randint(0, Xs.shape[0], int(Xt.shape[0]*0.1))
#         Xt_sample = Xt[idx]
#         dist_x2_sample = torch.ones(Xt_sample.shape[0]).float() / Xt_sample.shape[0]
#         dist_x2_sample.requires_grad = True
#         #init our regularizing params as parmeters for torch
#         reg1 = torch.nn.Parameter(torch.tensor(float(1)).float(), requires_grad=True)
#         reg2 = torch.nn.Parameter(torch.tensor(float(1)).float(), requires_grad=True)
#         #reg2 = torch.nn.Parameter(torch.tensor(self._kwargs['reg2']))

#         #init our transporter sinkhorn is the only supported method afaik

#         #self.transporter = EMDLaplaceTransport(reg_src=reg)

#         #init our optimizer
#         self.opt = torch.optim.Adam([reg1, reg2], lr=0.1)

#         self.lr_sch = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.99)
#         #init our loss function
#         self.loss_fn = torch.nn.MSELoss()
#         reg_history = []
#         state_history = []
#         #iter through the gradient descent
#         for i in np.arange(n_iter):
#             self.opt.zero_grad()
#             reg_history.append([np.copy(reg1.detach().numpy()), np.copy(reg2.detach().numpy())])
#             state_history.append(self.opt.state_dict())
#             #reinit the transporter with the new regularization
#             cost_matrix = ot.dist(Xt_sample, Xs)
#             #normalize cost matrix by max
#             cost_matrix = cost_matrix / torch.max(cost_matrix)

#             transp_map = ot.bregman.sinkhorn(dist_x2_sample, dist_x2, M=cost_matrix, reg=torch.pow(10,reg1), verbose=False)
#             #transport it back to the source
#             # perform standard barycentric mapping
#             transp = transp_map / torch.sum(transp_map, 1)[:, None]

#             # set nans to 0
#             transp[~ torch.isfinite(transp)] = 0

#             # compute transported samples
#             Xt_forward = torch.matmul(transp, Xs)

#             #transport it back to target
#             cost_matrix2 = ot.dist(Xt_forward, Xt)
#             #normalize cost matrix by max
#             cost_matrix2 = cost_matrix2 / torch.max(cost_matrix2)
#             transp_map2 = ot.bregman.sinkhorn(dist_x2_sample, dist_x1, M=cost_matrix2, reg=torch.pow(10,reg2), verbose=False)
#             #transport it back to the source
#             # perform standard barycentric mapping
#             transp2 = transp_map2 / torch.sum(transp_map2, 1)[:, None]


#             # set nans to 0
#             transp2[~ torch.isfinite(transp2)] = 0

#             # compute transported samples
#             Xt_backward = torch.matmul(transp2, Xt)
#             #compute the loss
#             loss = self.loss_fn(Xt_backward, Xt_sample)
#             #compute the gradient
#             #loss = torch.trace(torch.mm(transp2, cost_matrix2.T))
#             loss.backward()
#             #check if grad is nan and if so set to 0
#             # if torch.isnan(reg1.grad).any():
#             #     reg1.grad.zero_()
#             # if torch.isnan(reg2.grad).any():
                
#             #     reg2.grad.zero_()
#             #clip the gradient
#             torch.nn.utils.clip_grad_value_([reg1, reg2], 0.1)
#             #update the params
#             self.opt.step()
#             with torch.no_grad():
#                 if verbose:
#                     #reg.backward()
#                     plt.clf()
#                     xt_sample_np = Xt_sample.detach().numpy()
#                     plt.scatter(xt_sample_np[:, 0], xt_sample_np[:, 1], c='r', s=1)
#                     plt.scatter(Xt_backward.detach().numpy()[:, 0], Xt_backward.detach().numpy()[:, 1], c='b', s=1)
#                     plt.pause(0.100)
#                     print('iteration: ', i, 'loss: ', loss.item(), 'reg1: ', reg1.detach().numpy(), 'reg2: ', reg2.detach().numpy(), 'lr: ', self.lr_sch.get_lr())
#                     #if the reg becomes nan we want to reset and try again

#                     if (np.isnan(reg1.detach().numpy()) or np.isnan(reg2.detach().numpy())):
#                         #reinit the reg params
#                         self.lr_sch.step()
#                         #reset the nan reg params
#                         if np.isnan(reg1.detach().numpy()):
#                             reg1 = torch.nn.Parameter(torch.tensor(float(reg_history[-2][0])).float(), requires_grad=True)
#                         if np.isnan(reg2.detach().numpy()):
#                             reg2 = torch.nn.Parameter(torch.tensor(float(reg_history[-2][1])).float(), requires_grad=True)
#                         self.opt = torch.optim.Adam([reg1, reg2], lr=self.lr_sch.get_lr()[0])
#                         print('reset reg')
#                         if self.lr_sch.get_lr()[0] < 1e-9:
#                             self.opt = torch.optim.Adam([reg1, reg2], lr=1)
#                             self.lr_sch = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.7)
#                         else:
#                             self.opt = torch.optim.Adam([reg1, reg2], lr=self.lr_sch.get_lr()[0])
#         #once we complete the loop, get the best kwargs and init the transporter
#         reg_back_numpy = reg_history[-1][1]
#         #for now directly inst and assign the transporter
#         self.transporter = ot.da.SinkhornTransport(reg_e=reg_back_numpy)










#  def tune_grad_descent_emp(self, Xs, Xt, Ys=None, Yt=None, n_iter=500, verbose=True):
#         #for certain methods, we can use gradient descent to find the best params
        
#         #put our data into torch
#         Xs = torch.from_numpy(Xs).float()
#         Xt = torch.from_numpy(Xt).float()

#         dist_x1 = torch.ones( Xt.shape[0]).float() / Xt.shape[0]
#         dist_x2 = torch.ones(Xs.shape[0]).float() / Xs.shape[0]

#         dist_x1.requires_grad = True
#         dist_x2.requires_grad = True

#         #grab a random sample
#         idx = np.random.randint(0, Xs.shape[0], int(Xt.shape[0]*0.1))
#         Xt_sample = Xt[idx]
#         dist_x2_sample = torch.ones(Xt_sample.shape[0]).float() / Xt_sample.shape[0]
#         dist_x2_sample.requires_grad = True
#         #init our regularizing params as parmeters for torch
#         reg1 = torch.nn.Parameter(torch.tensor(float(1)).float(), requires_grad=True)
#         reg2 = torch.nn.Parameter(torch.tensor(float(1)).float(), requires_grad=True)
#         #reg2 = torch.nn.Parameter(torch.tensor(self._kwargs['reg2']))

#         #init our transporter sinkhorn is the only supported method afaik

#         #self.transporter = EMDLaplaceTransport(reg_src=reg)

#         #init our optimizer
#         self.opt = torch.optim.SGD([reg1, reg2], lr=10)

#         self.lr_sch = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.7)
#         #init our loss function
#         self.loss_fn = torch.nn.MSELoss()
#         reg_history = []
#         state_history = []
#         #iter through the gradient descent
#         for i in np.arange(n_iter):
#             self.opt.zero_grad()
#             reg_history.append([np.copy(reg1.detach().numpy()), np.copy(reg2.detach().numpy())])
#             state_history.append(self.opt.state_dict())
#             #reinit the transporter with the new regularization
#             cost_matrix = ot.dist(Xt_sample, Xs)
#             transp_map = ot.bregman.empirical_sinkhorn(Xt_sample, Xs, reg1, verbose=False)
#             #transport it back to the source
#             # perform standard barycentric mapping
#             transp = transp_map / torch.sum(transp_map, 1)[:, None]

#             # set nans to 0
#             transp[~ torch.isfinite(transp)] = 0

#             # compute transported samples
#             Xt_forward = torch.matmul(transp, Xs)

#             #transport it back to target
#             cost_matrix2 = ot.dist(Xt_forward, Xt)
#             transp_map2 = ot.bregman.empirical_sinkhorn(Xt_forward, Xt, reg2, verbose=False)
#             #transport it back to the source
#             # perform standard barycentric mapping
#             transp2 = transp_map2 / torch.sum(transp_map2, 1)[:, None]


#             # set nans to 0
#             transp2[~ torch.isfinite(transp2)] = 0

#             # compute transported samples
#             Xt_backward = torch.matmul(transp2, Xt)
#             #compute the loss
#             loss = self.loss_fn(Xt_backward, Xt_sample)
#             #compute the gradient
#             #loss = torch.trace(torch.mm(transp2, cost_matrix2.T))
#             loss.backward()
#             #update the params
#             self.opt.step()
#             with torch.no_grad():
#                 if verbose:
#                     #reg.backward()
#                     plt.clf()
#                     xt_sample_np = Xt_sample.detach().numpy()
#                     plt.scatter(xt_sample_np[:, 0], xt_sample_np[:, 1], c='r', s=1)
#                     plt.scatter(Xt_backward.detach().numpy()[:, 0], Xt_backward.detach().numpy()[:, 1], c='b', s=1)
#                     plt.pause(0.100)
#                     print('iteration: ', i, 'loss: ', loss.item(), 'reg1: ', reg1.detach().numpy(), 'reg2: ', reg2.detach().numpy(), 'lr: ', self.lr_sch.get_lr())
#                     #if the reg becomes nan we want to reset and try again
#                     if np.isnan(reg1.detach().numpy()):
#                         #reinit the reg params
#                         self.lr_sch.step()
#                         reg1 = torch.nn.Parameter(torch.tensor(float(reg_history[-1][0])).float(), requires_grad=True)
#                         reg2 = torch.nn.Parameter(torch.tensor(float(reg_history[-1][1])).float(), requires_grad=True)

#                         self.opt = torch.optim.Adam([reg1, reg2], lr=self.lr_sch.get_lr()[0])
#                         #if lr is really low warm restart the lr scheduler
#                         if self.lr_sch.get_lr()[0] < 1e-9:
#                             self.lr_sch = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.7)
#                         print('reset reg')
#                         #self.lr_sch.step()
#         #get the best transporter

# def _bidirectional_tune(self, Xs, Xt, Ys=None, Yt=None, n_iter=20, n_jobs=-1, verbose=True):
#         """ Tune the parameters of the OTDA based transporter to the datasets provided.
#         Tunes the reg parameters and the norm of the data sets. Currently supports using an error function base on
#         the mse between randomly skewed data and Xt.
#         Supports paralellism via n_jobs. The total number of points queried is (n_iter x n_jobs)
#         Uses nevergrad as optimizer backend. TODO// allow user to select optimizer

#         Args:
#             Xs (numpy array): _description_
#             Xt (numpy array): _description_
#             Ys (_type_, optional): _description_. Defaults to None.
#             Yt (_type_, optional): _description_. Defaults to None.
#             n_iter (int, optional): _description_. Defaults to 10.
#             n_jobs (int, optional): _description_. Defaults to -1.
#             method (str, optional): _description_. Defaults to 'unsuper-multi'.
#             verbose (bool, optional): _description_. Defaults to True.

#         Returns:
#             _type_: _description_
#         """

#         #if the user does not specify just use the default values
#         if n_jobs == -1:
#             n_jobs = multiprocessing.cpu_count()

#         #init a nevergrad optimization
#         self.opt = ng.optimizers.Portfolio(self.param_dict, budget=n_iter*n_jobs, num_workers=n_jobs,)
#         with warnings.catch_warnings():#Catch the sinkhorn warnings
#             warnings.simplefilter("ignore")
#             for i in np.arange(n_iter):

#                 #get the current params
#                 param_list = []
#                 for n in np.arange(n_jobs):
#                     param_list.append(self.opt.ask())
                
                
#                 #get the current score
#                 if self.flexible_transporter:
#                     #get the current transporter
                    
#                     score = joblib.Parallel(n_jobs=n_jobs,  prefer="threads", require='sharedmem', verbose=100)(joblib.delayed(_inner_tune_back_and_forth)(Xs, Xt, Ys, Yt, opt_kwargs=p.value) for p in param_list)
#                 else:
#                     transporter = self.inittransporter
#                     score = joblib.Parallel(n_jobs=n_jobs,  prefer="threads", require='sharedmem', mmap_mode=None, verbose=100)(joblib.delayed(_inner_tune_back_and_forth)(Xs, Xt, Ys, Yt, transporter=transporter, opt_kwargs=p.value) for p in param_list)
#                 #update the nevergrad params
#                 for p, e in zip(param_list, score):
#                     self.opt.tell(p, e)
#                 #self.opt.tell(params, score)
#                 if verbose:
#                     print('iteration: ', i, 'score: ', np.amin(score))
#         #get the best transporter
#         print("best kwargs:")
#         print(self.opt.recommend().value)
        
#         best_kwargs = self._to_kwarg_dict(transporter=None, kwargs=self.opt.recommend().value)
#         self.transporter = self.inittransporter(**best_kwargs,  log=True)
#         self.best_ = best_kwargs
#         return self