import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    logger.warning("h5py not installed, NWB loading will not work")

import pandas as pd


def loadNWB(file_path, return_obj=False, old=False):
    """Loads the nwb object and returns three arrays dataX, dataY, dataC and optionally the object.
    same input / output as loadABF for easy pipeline inclusion

    Args:
        file_path (str): [description]
        return_obj (bool, optional): return the NWB object to access various properites. Defaults to False.
        old (bool, optional): use the old indexing method, uneeded in most cases. Defaults to False.

    Returns:
        dataX: time (should be seconds)
        dataY: voltage (should be mV)
        dataC: current (should be pA)
        dt: time step (should be seconds)
    """    
   
    if old:
        nwb = old_nwbFile(file_path)
    else:
        nwb = nwbFile(file_path)
    
    dt = np.reciprocal(float(nwb.rate))  # seconds, using actual sampling rate from file

    if isinstance(nwb.dataX, np.ndarray)==False:
        dataX = np.asarray(nwb.dataX, dtype=np.dtype('O')) ##Assumes if they are still lists its due to uneven size
        dataY = np.asarray(nwb.dataY, dtype=np.dtype('O')) #Casts them as numpy object types to deal with this
        dataC = np.asarray(nwb.dataC, dtype=np.dtype('O'))
    else:
        dataX = nwb.dataX #If they are numpy arrays just pass them
        dataY = nwb.dataY
        dataC = nwb.dataC

    if return_obj:
        return dataX, dataY, dataC, dt, nwb
    else:
        return dataX, dataY, dataC, dt


# A simple class to load the nwb data quick and easy
##Call like nwb = nwbfile('test.nwb')
##Sweep data is then located at nwb.dataX, nwb.dataY, nwb.dataC (for stim)
class old_nwbFile(object):
    """Loads NWB files using the older acquisition key layout (no 'timeseries' subgroup)."""

    def __init__(self, file_path):
        with h5py.File(file_path,  "r") as f:
            ##Load some general properities
            sweeps = list(f['acquisition'].keys()) ##Sweeps are stored as keys
            self.sweepCount = len(sweeps)
            self.rate = dict(f['acquisition'][sweeps[0]]['starting_time'].attrs.items())['rate']
            self.sweepYVars = dict(f['acquisition'][sweeps[0]]['data'].attrs.items())
            ##Load the response and stim
            dataY = []
            dataX = []
            dataC = []
            for sweep in sweeps:
                ##Load the response and stim
                data_space_s = 1/(dict(f['acquisition'][sweeps[0]]['starting_time'].attrs.items())['rate'])
                temp_dataY = np.asarray(f['acquisition'][sweep]['data'][()])
                temp_dataX = np.cumsum(np.hstack((0, np.full(temp_dataY.shape[0]-1,data_space_s))))
                try:
                    temp_dataC = np.asarray(f['stimulus']['presentation'][sweep]['data'][()])
                    dataC.append(temp_dataC)
                except KeyError:
                    logger.debug(f"No stimulus data found for sweep {sweep}")
                dataY.append(temp_dataY)
                dataX.append(temp_dataX)
            try:
                ##Try to vstack assuming all sweeps are same length
                self.dataX = np.vstack(dataX)
                self.dataY = np.vstack(dataY)
                self.dataC = np.vstack(dataC) if dataC else np.array([])
            except ValueError:
                #Just leave as lists if sweeps have different lengths
                self.dataX = dataX
                self.dataY = dataY
                self.dataC = dataC
        return




class nwbFile(object):
    """Loads NWB files using the newer acquisition/timeseries key layout."""

    def __init__(self, file_path):
        with h5py.File(file_path,  "r") as f:
            ##Load some general properities
            sweeps = list(f['acquisition']['timeseries'].keys()) ##Sweeps are stored as keys
            self.sweepCount = len(sweeps)
            try:
                self.sweepCVars = dict(f['stimulus']['presentation'][sweeps[-1]]['data'].attrs.items())
            except KeyError:
                self.sweepCVars = None
            ## Find the index's with long square
            index_to_use = []
            for key in sweeps: 
                sweep_dict = f['stimulus']['presentation'][key]['aibs_stimulus_name'][()]
                if check_stimulus(sweep_dict):
                    index_to_use.append(key) 

            # Get sampling rate from first usable sweep
            self.rate = dict(f['acquisition']['timeseries'][index_to_use[0]]['starting_time'].attrs.items())['rate'] if index_to_use else 50000

            dataY = []
            dataX = []
            dataC = []
            for sweep in index_to_use:
                ##Load the response and stim
                data_space_s = 1/(dict(f['acquisition']['timeseries'][sweep]['starting_time'].attrs.items())['rate'])
                try:
                    bias_current = f['acquisition']['timeseries'][sweep]['bias_current'][()]
                    if np.isnan(bias_current):
                        bias_current = 0
                except KeyError:
                    bias_current = 0
                temp_dataY = np.asarray(f['acquisition']['timeseries'][sweep]['data'][()])
                temp_dataX = np.cumsum(np.hstack((0, np.full(temp_dataY.shape[0]-1,data_space_s))))
                temp_dataC = np.asarray(f['stimulus']['presentation'][sweep]['data'][()]) #+ (bias_current * 1e+12) #in pA => A
                dataY.append(temp_dataY)
                dataX.append(temp_dataX)
                dataC.append(temp_dataC)
            try:
                ##Try to vstack assuming all sweeps are same length
                self.dataX = np.vstack(dataX)
                self.dataC = np.vstack(dataC)
                self.dataY = np.vstack(dataY)
            except ValueError:
                #Just leave as lists if sweeps have different lengths
                self.dataX = dataX
                self.dataC = dataC
                self.dataY = dataY
        return

class stim_names:
    """Container for stimulus name inclusion/exclusion patterns."""
    stim_inc = ['Long', '1000']
    stim_exc = ['rheo', 'Rf50_']
    def __init__(self):
        self.stim_inc = stim_names.stim_inc
        self.stim_exc = stim_names.stim_exc
        return

global_stim_names = stim_names()
def check_stimulus(stim_desc):
    try:
        stim_desc_str = stim_desc.decode()
    except (AttributeError, UnicodeDecodeError):
        stim_desc_str = str(stim_desc)
    include_s = np.any([x in stim_desc_str for x in global_stim_names.stim_inc])
    exclude_s = np.invert(np.any([x in stim_desc_str for x in global_stim_names.stim_exc]))
    return np.logical_and(include_s, exclude_s)
