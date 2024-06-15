import pickle as pkl
import os

folder = os.path.dirname(os.path.abspath(__file__))
#load the data
MMS_DATA = pkl.load(open(folder + '/mms_data.pkl', 'rb'))
