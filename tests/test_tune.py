import patchOTDA.domain_adapt as pOTDA
from ot.datasets import make_2D_samples_gauss, make_data_classif
import numpy as np
import cProfile
import matplotlib.pyplot as plt


pOTDA.TIMEOUT = None # Disable timeout for testing
def profile_fund_function():
    Xs = make_2D_samples_gauss(n=1000, m=(0,0), sigma=np.full((2,2), 0.5))
    Xt = make_2D_samples_gauss(n=1500, m=(2.5,.5), sigma=np.full((2,2), 3.5))
    with cProfile.Profile() as pr:
        pOTDA._tune_transporter(Xs=Xs, Xt=Xt, Ys=None, Yt=None, transporter=pOTDA.unbalancedFUGWTransporter)
        stats = pr.print_stats(sort='cumtime')

        print(stats)

    

def test_tune_unsuper():
    # Create a PatchClampOTDA object
    p = pOTDA.PatchClampOTDA(flexible_transporter=False)
    #make a dataset

    Xs = make_2D_samples_gauss(n=1000, m=(0,0), sigma=np.array([[1, 0], [0, 1]]))
    Xt = make_2D_samples_gauss(n=1500, m=(2.5,.5), sigma=np.array([[1, -.8], [-.8, 1]]))
    
    p.tune(Xs=Xs, Xt=Xt, n_jobs=10, n_iter=200, method="unidirectional", verbose=True)


    Xs_shifted = p.fit_transform(Xs, Xt)

    plt.scatter(Xs[:,0], Xs[:,1], label="Xs")
    plt.scatter(Xt[:,0], Xt[:,1], label="Xt")
    plt.scatter(Xs_shifted[:,0], Xs_shifted[:,1], label="Xs_shifted")
    plt.legend()
    #plt.show()

def test_tune():
    #test supervised
    Xs, Ys = make_data_classif(dataset="3gauss", n=100, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=1500, nz=0.5)

    p = pOTDA.PatchClampOTDA('SinkhornLpl1Transport', flexible_transporter=False)

    p.tune(Xs=Xs, Xt=Xt, Ys=Ys, Yt=Yt, n_jobs=1, n_iter=200, method="unidirectional", supervised=True, verbose=True)

    Xs_shifted = p.fit_transform(Xs, Xt, Ys, Yt)

    plt.scatter(Xs[:,0], Xs[:,1], c=Ys, label="Xs", marker="x")
    plt.scatter(Xt[:,0], Xt[:,1], c=Yt, label="Xt", alpha=0.1)
    plt.scatter(Xs_shifted[:,0], Xs_shifted[:,1], c=Ys, label="Xs_shifted")
    plt.legend()
    plt.show()
    return True
        

if __name__=="__main__":
    profile_fund_function()
    test_tune_unsuper()
    print("test_tune passed")