"""
Bundled reference datasets for patchOTDA.

Loads the Allen Institute MMS (MapMySpikes) reference data from a bundled pickle file.
Access via the ``MMS_DATA`` dict with keys like ``'CTKE_M1'``, ``'VISp_Viewer'``, ``'joint_feats'``.

.. warning::
    The data is loaded from a pickle file. Only use the bundled file shipped with
    this package — never replace it with untrusted data.
"""
import pickle as pkl
from pathlib import Path

_data_path = Path(__file__).parent / 'mms_data.pkl'

with open(_data_path, 'rb') as _f:
    MMS_DATA = pkl.load(_f)
