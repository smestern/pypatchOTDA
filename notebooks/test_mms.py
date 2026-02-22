# %% [markdown]
# Very WIP notebooks to help finish up the project.

# %%
import os
os.chdir("/home/smestern/patchotda/")
import streamlit as st
import pandas as pd
import numpy as np
#sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
#import XGBoost as xgb
#other
import altair as alt
import time
import zipfile
from .utils import MMS_DATA, USER_DATA, EXAMPLE_DATA_, REF_DATA_, VISp_MET_nodes, VISp_T_nodes, filter_MMS, find_outlier_idxs, param_grid_from_dict, select_by_col, not_select_by_col
import os
from patchOTDA.external import skada
from patchOTDA.domainAdapt import PatchClampOTDA
from functools import partial
import ot.da
from ot.backend import get_backend
import umap
from streamlit_tree_select import tree_select
import xgboost as xgb

from streamlit_app import CLASS_MODELS, HICLASS_METHOD

labels="VISp_T"
hiclass_method = "LocalClassifierPerNode"
class_model = "Random Forest"


# %%
ref_data = MMS_DATA['CTKE_M1']
ref_data_ephys, ref_data_meta = ref_data['ephys'], ref_data['meta']
Xs, Ys = ref_data_ephys, ref_data_meta[[labels+"_1_en", labels+"_2_en", labels+"_3_en"]]
Ys = Ys.to_numpy()

# %%
Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(Xs, Ys, test_size=0.2, random_state=42)

# %%
impute_and_scale = Pipeline([('imputer', KNNImputer()), ('scaler', StandardScaler())])
Xs_train = impute_and_scale.fit_transform(Xs_train)
Xs_test = impute_and_scale.transform(Xs_test)

# %%

CLASS_MODELS[class_model]['params']





class dummy_cv_grid(HICLASS_METHOD[hiclass_method]): #this is a hack to get around the fact that the gridsearchcv does not like the LocalClassifierPerNode
    def __init__(self, **kwargs):
        super().__init__(CLASS_MODELS[class_model]['model'](**kwargs))
    def set_params(self, **params):
        return super().__init__(CLASS_MODELS[class_model]['model'](**params))





grid_params = param_grid_from_dict(CLASS_MODELS[class_model]['params'])

searcher = GridSearchCV(dummy_cv_grid(), param_grid=grid_params, cv=5, n_jobs=-1, verbose=1)
searcher.fit(Xs_test, Ys_test[:,-1].astype(np.int32))


# %%


model = HICLASS_METHOD[hiclass_method](CLASS_MODELS[class_model]['model'](),n_jobs=-1)



# %%
model.fit(Xs_train, Ys_train)

# %%
Ys_pred = model.predict(Xs_test)
Ys_pred_train = model.predict(Xs_train)

#test the 3 levels of the model
dict_results = {"Train":[], "Test":[]}
for level in [0,1,2]:
    dict_results[f"Train"].append(accuracy_score(Ys_train[:,level].astype(np.int32), Ys_pred_train[:,level].astype(np.int32)))
    dict_results[f"Test"].append(accuracy_score(Ys_pred[:,level].astype(np.int32), Ys_test[:,level].astype(np.int32)))


# %%
print(dict_results)


