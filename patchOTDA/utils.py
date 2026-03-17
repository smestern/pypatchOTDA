"""
Utility functions and configuration constants for patchOTDA.

Provides DataFrame helpers, MMS label-tree builders, hyperparameter grid
construction, outlier detection, and pre-built model/classifier registries.
"""

import logging

import numpy as np
import ot
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .datasets import MMS_DATA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — guarded per project convention
# ---------------------------------------------------------------------------

try:
    from hiclass import (
        LocalClassifierPerNode,
        LocalClassifierPerParentNode,
        LocalClassifierPerLevel,
    )
except ImportError:
    logger.warning(
        "hiclass not installed — HICLASS_METHOD will be empty. "
        "Install with: pip install hiclass"
    )
    LocalClassifierPerNode = None
    LocalClassifierPerParentNode = None
    LocalClassifierPerLevel = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning(
        "xgboost not installed — XGBoost will not be available in CLASS_MODELS. "
        "Install with: pip install xgboost"
    )

try:
    from .external import skada as _skada
except Exception:
    _skada = None
    logger.warning(
        "skada wrappers could not be loaded — JDOTC will not be available in MODELS."
    )

# ---------------------------------------------------------------------------
# Dataset name lists
# ---------------------------------------------------------------------------

EXAMPLE_DATA_ = ['Query1', 'Query2', 'Query3', 'CTKE_M1', 'VISp_Viewer']

REF_DATA_ = ['CTKE_M1', 'VISp_Viewer']

# ---------------------------------------------------------------------------
# Hierarchical label trees (built from MMS_DATA at import time)
# ---------------------------------------------------------------------------

VISp_MET_nodes = [
    {'label': 'inhibitory', 'value': 'inhibitory', 'color': '#0000FF', 'children': []},
]

VISp_T_nodes = [
    {'label': 'excitatory', 'value': 'excitatory', 'color': '#FF0000', 'children': []},
    {'label': 'inhibitory', 'value': 'inhibitory', 'color': '#0000FF', 'children': []},
]


def _build_met_nodes():
    """Parse MMS_DATA to populate VISp_MET_nodes hierarchy."""
    for data in REF_DATA_:
        if data not in MMS_DATA:
            continue
        mms_temp = MMS_DATA[data]['meta']
        if 'VISp Viewer MET type' not in mms_temp.columns:
            continue
        labels = mms_temp['VISp Viewer MET type'].fillna('unk').values
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_split = label.split('-')
            if len(label_split) == 1:
                continue
            parent = label_split[0]
            child = '_'.join(label_split)
            if parent not in [x['label'] for x in VISp_MET_nodes[0]['children']]:
                VISp_MET_nodes[0]['children'].append(
                    {'label': parent, 'value': parent, 'color': '#00FF00', 'children': []})
            idx = [x['label'] for x in VISp_MET_nodes[0]['children']].index(parent)
            if child not in [x['label'] for x in VISp_MET_nodes[0]['children'][idx]['children']]:
                VISp_MET_nodes[0]['children'][idx]['children'].append(
                    {'label': child, 'value': child, 'color': '#00FF00'})


def _build_t_nodes():
    """Parse MMS_DATA to populate VISp_T_nodes hierarchy."""
    for data in REF_DATA_:
        if data not in MMS_DATA:
            continue
        mms_temp = MMS_DATA[data]['meta']
        if 'VISp Viewer T type' not in mms_temp.columns:
            continue
        labels = mms_temp['VISp Viewer T type'].fillna('unk').values
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_split = label.split(' ')
            if len(label_split) == 1:
                continue
            if label_split[0] in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L6b', 'L2/3']:
                parent = label_split[0] + '_' + label_split[1]
                idx0 = 0
                child1 = '_'.join(label_split[0:3])
                child2 = '_'.join(label_split[0:4])
            else:
                parent = label_split[0]
                idx0 = 1
                child1 = '_'.join(label_split[0:2])
                child2 = '_'.join(label_split[0:3])

            if parent not in [x['label'] for x in VISp_T_nodes[idx0]['children']]:
                VISp_T_nodes[idx0]['children'].append(
                    {'label': parent, 'value': parent, 'color': '#00FF00', 'children': []})

            if child1 == parent:
                continue
            idx1 = [x['label'] for x in VISp_T_nodes[idx0]['children']].index(parent)
            if child1 not in [x['label'] for x in VISp_T_nodes[idx0]['children'][idx1]['children']]:
                VISp_T_nodes[idx0]['children'][idx1]['children'].append(
                    {'label': child1, 'value': child1, 'color': '#00FF00', 'children': []})

            if child2 == child1:
                continue
            idx2 = [x['label'] for x in VISp_T_nodes[idx0]['children'][idx1]['children']].index(child1)
            if child2 not in [x['label'] for x in VISp_T_nodes[idx0]['children'][idx1]['children'][idx2]['children']]:
                VISp_T_nodes[idx0]['children'][idx1]['children'][idx2]['children'].append(
                    {'label': child2, 'value': child2, 'color': '#00FF00'})


def _build_label_encoders():
    """Fit LabelEncoders for each reference dataset's hierarchical labels."""
    for data in REF_DATA_:
        if data not in MMS_DATA:
            continue
        mms_temp = MMS_DATA[data]['meta']

        if 'VISp Viewer MET type' in mms_temp.columns:
            MMS_DATA[data]['meta']['VISp_MET_3'] = MMS_DATA[data]['meta']['VISp_MET_2']
            mms_temp = MMS_DATA[data]['meta']
            for x in [1, 2, 3]:
                labels = mms_temp[f'VISp_MET_{x}'].fillna('unk').values
                unique_labels = np.unique(labels)
                le = LabelEncoder()
                le.fit(unique_labels)
                MMS_DATA[data][f'VISp_MET_LE_{x}'] = le
                MMS_DATA[data]['meta'][f'VISp_MET_{x}_en'] = le.transform(labels)

        if 'VISp Viewer T type' in mms_temp.columns:
            for x in [1, 2, 3]:
                labels = mms_temp[f'VISp_T_{x}'].fillna('unk').values
                unique_labels = np.unique(labels)
                le = LabelEncoder()
                le.fit(unique_labels)
                MMS_DATA[data][f'VISp_T_LE_{x}'] = le
                MMS_DATA[data]['meta'][f'VISp_T_{x}_en'] = le.transform(labels)


# Run builders at import time (matches original utils.py behavior)
if MMS_DATA is not None:
    try:
        _build_met_nodes()
        _build_t_nodes()
        _build_label_encoders()
    except Exception:
        logger.warning("Failed to build label trees / encoders from MMS_DATA.", exc_info=True)

# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

HICLASS_METHOD = {}
if LocalClassifierPerNode is not None:
    HICLASS_METHOD = {
        'LocalClassifierPerNode': LocalClassifierPerNode,
        'LocalClassifierPerParentNode': LocalClassifierPerParentNode,
        'LocalClassifierPerLevel': LocalClassifierPerLevel,
    }

MODELS = {
    'EMDLaplace (EMD based distance transport) - Unsupervised': {
        'model': ot.da.EMDLaplaceTransport,
        'params': {
            'reg_lap': (0., 10., 10.),
            'reg_src': (0., 10., 0.9),
            'norm': ['median', None, 'max'],
            'verbose': False,
        },
        'Description': '',
    },
    'UnbalancedSinkhornTransport (Optimal Transport) - Unsupervised - unevenly Sampled': {
        'model': ot.da.UnbalancedSinkhornTransport,
        'params': {
            'reg_e': (0., 2., 0.1),
            'reg_m': (0., 2., 0.1),
            'max_iter': (10, 10000, 1000),
            'norm': [None, 'median', 'max'],
            'verbose': False,
        },
        'Description': '',
    },
}

if _skada is not None and hasattr(_skada, 'JDOTC'):
    MODELS['JDOT (Joint Distribution Optimal Transport) - Semisupervised - unevenly Sampled'] = {
        'model': _skada.JDOTC,
        'params': {
            'alpha': (0., 10., 0.1),
            'n_iter_max': (10, 10000, 1000),
        },
        'Description': '',
    }

CLASS_MODELS = {
    'Random Forest': {
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': (1, 1000, 100, 3),
            'max_depth': [50, None, 3, 5, 25, 100, 200],
            'min_samples_split': (2, 100, 2, 3),
            'min_samples_leaf': (1, 100, 2),
            'min_impurity_decrease': (0.0, 1., 0.0, 3),
        },
        'Description': '',
    },
    'Logistic Regression': {
        'model': LogisticRegression,
        'params': {
            'C': (0., 1., 1.0, 4),
            'penalty': ['l2', 'l1', 'elasticnet'],
            'solver': ['saga', 'sag'],
            'l1_ratio': [None, 0.1, 0.3, 0.5, 0.75, 0.9],
        },
        'Description': '',
    },
    'SVM': {
        'model': SVC,
        'params': {
            'C': (0., 1., 1.0, 4),
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        },
        'Description': '',
    },
}

if xgb is not None:
    CLASS_MODELS['XGBoost'] = {
        'model': xgb.XGBClassifier,
        'params': {
            'n_estimators': (1, 1000, 100, 3),
            'max_depth': (1, 100, 2, 4),
            'learning_rate': (0.0, 1.0, 0.1, 4),
        },
        'Description': '',
    }

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def select_by_col(df, cols):
    """Return DataFrame columns whose names intersect with *cols*."""
    if isinstance(cols, str):
        cols = [cols]
    union = np.intersect1d(cols, df.columns)
    return df[union]


def not_select_by_col(df, cols):
    """Return DataFrame columns NOT in *cols*."""
    if isinstance(cols, str):
        cols = [cols]
    set_diff = np.setdiff1d(df.columns, cols)
    return df[set_diff]


def filter_MMS(data, label_query, query):
    """Return row indices of MMS_DATA[data] where *label_query* column contains any of query['checked']."""
    mms_temp = MMS_DATA[data]['meta']
    selected_labels = query['checked']
    if label_query not in mms_temp.columns:
        return None
    labels = mms_temp[label_query].fillna('unk').values
    idx = np.nonzero([x in selected_labels for x in labels])[0]
    return idx


def param_grid_from_dict(param_dict):
    """Convert a hyperparameter spec dict to a grid suitable for GridSearchCV.

    Tuple values are expanded via ``np.linspace``:
    - ``(min, max, default, n_steps)`` -> ``linspace(min, max, n_steps)``
    - ``(min, max, default)`` -> ``linspace(min, max, 4)``
    - ``(min, max)`` -> ``linspace(min, max, 4)``
    Bool values become ``[True, False]``.  Lists are converted to arrays.
    """
    return_dict = {}
    for key, value in param_dict.items():
        if isinstance(value, tuple):
            if len(value) == 4:
                return_dict[key] = np.linspace(value[0], value[1], value[-1]).astype(type(value[0]))
            else:
                return_dict[key] = np.linspace(value[0], value[1], 4).astype(type(value[0]))
        elif isinstance(value, bool):
            return_dict[key] = np.array([True, False])
        elif isinstance(value, list):
            return_dict[key] = np.array(value)
    return return_dict


def find_outlier_idxs(X, n_outliers=10):
    """Return indices of outliers in *X* using IsolationForest (contamination=0.1)."""
    scale_and_imputer = Pipeline([
        ('impute', SimpleImputer()),
        ('scaler', StandardScaler()),
    ])
    x = scale_and_imputer.fit_transform(X)
    clf = IsolationForest(contamination=0.1)
    clf.fit(x)
    return np.where(clf.predict(x) == -1)[0]
