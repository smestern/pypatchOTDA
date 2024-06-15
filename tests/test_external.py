from patchOTDA.nn import uniOTtab
import pandas as pd
import numpy as np
from patchOTDA.nn.uniood.configs import parser
from patchOTDA.external import skada
from ot.datasets import make_2D_samples_gauss, make_data_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

path = "/home/smestern/Downloads/MapMySpikes_data_PUBLIC final.xlsx"

df = pd.read_excel(path, sheet_name="CTKE_M1", index_col=0)
def test_uniOTtab_dataset():
    #load the data 
    df = pd.read_excel(path, sheet_name="CTKE_M1", index_col=0)

    features = pd.read_excel(path, sheet_name="Ephys_Features")
    df_ephys, df_meta = select_by_col(df, features['Parameter name'].values), not_select_by_col(df, features['Parameter name'].values)
    #create the dataset
    dataset = uniOTtab.TabularDataset(source_domain=df, target_domain=df, n_share=100, n_source_private=10)

    pass

def test_uniOTtab_model():

    #args = parser.parse_args()

    model = uniOTtab.UniOTtab()

    #test supervised
    Xs, Ys = make_data_classif(dataset="3gauss", n=10000, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=15000, nz=0.5)

    #skew the data a bit
    Xt[:,0] = Xt[:,0] + 4
    Xt[:,1] = Xt[:,1] + 4
    #Xs = Xs + np.random.normal(0, 4, Xs.shape)


    plt.scatter(Xs[:, 0], Xs[:, 1], c=Ys, marker='x')
    plt.scatter(Xt[:, 0], Xt[:, 1], c=Yt, alpha=0.01)
    
    #make a classification model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(Xs, Ys)
    scores = {"source": rf.score(Xs, Ys), "target": rf.score(Xt, Yt)}


    model.fit(Xs, Xt, Ys, Yt, n_share=4, n_source_private=0, num_internal_cluster=4, max_iter=150, base_lr=1e-3)
    out = model.predict(Xt)
    out2 = model.predict(Xs)

    
    scores['source_da'] = accuracy_score(Ys, out2)
    scores['target_da'] = accuracy_score(Yt, out)

    embed = model.transform(Xs)
    embed2 = model.transform(Xt)
    plt.figure()
    plt.scatter( embed2[:, 0],  embed2[:, 1], marker='x', c=Yt)
    plt.scatter( embed[:, 0],  embed[:, 1], alpha=0.1, c=Ys)
    #plt.show()
    print(scores)

    return


def test_skada():
    jdot = skada.JDOTC(n_iter_max=int(1e4))
    #test supervised
    Xs, Ys = make_data_classif(dataset="3gauss", n=1000, nz=0.5)
    Xt, Yt = make_data_classif(dataset="3gauss2", n=1500, nz=0.5)

    #skew the data a bit
    Xt[:,0] = Xt[:,0] + 4
    Xt[:,1] = Xt[:,1] + 4

    #make a classification model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(Xs, Ys)
    scores = {"source": rf.score(Xs, Ys), "target": rf.score(Xt, Yt)}

    jdot.fit(Xs, Xt, Ys, Yt)
    #out = jdot.predict(Xs)
    out2 = jdot.predict(Xt)
    scores['target_da'] = accuracy_score(Yt, out2)
    #embed = jdot.transform(Xs)

    #plt.scatter( embed2[:, 0],  embed2[:, 1], marker='x', c=Yt)
    #plt.scatter( embed[:, 0],  embed[:, 1], alpha=0.1, c=Ys)

    return





def select_by_col(df, cols):
    if isinstance(cols, str):
        cols = [cols]
    df_cols = df.columns
    union = np.intersect1d(cols, df_cols)
    return df[union]

def not_select_by_col(df, cols):
    if isinstance(cols, str):
        cols = [cols]
    df_cols = df.columns
    set_diff = np.setdiff1d(df_cols, cols)
    return df[set_diff]




if __name__ == "__main__":
    test_skada()
    #test_uniOTtab_model()