"""
Bundled reference datasets for patchOTDA.

Loads the Allen Institute MMS (MapMySpikes) reference data from bundled CSV files.
Access via the ``MMS_DATA`` dict with keys like ``'CTKE_M1'``, ``'VISp_Viewer'``, ``'joint_feats'``.
"""
import json
from pathlib import Path

import pandas as pd

_data_dir = Path(__file__).parent

with open(_data_dir / 'mms_metadata.json', 'r') as _f:
    _meta = json.load(_f)

#Here we used to have pickled dataframes, but it broke everytime Pandas updated. 
# So now we have CSVs, which are more robust to changes in Pandas versions. The code is a bit more verbose, but it works.

MMS_DATA = {'joint_feats': _meta['joint_feats']}

for _name, _files in _meta['datasets'].items():
    MMS_DATA[_name] = {
        k: pd.read_csv(_data_dir / v, index_col=0)
        for k, v in _files.items()
    }
