from pyedflib import highlevel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

path = "PN00-2.edf"  ## Original EEG Data
signals, signal_headers, header = highlevel.read_edf(path)
n = len(signals)

print(signals.T.shape)

def get_sec(t1,t2):
    td1 = timedelta(hours = t1[0],minutes = t1[1],seconds = t1[2])
    td2 = timedelta(hours = t2[0],minutes = t2[1],seconds = t2[2])
    r = int(td2.total_seconds() - td1.total_seconds())*512
    return r

df = pd.DataFrame(signals.T[get_sec([19,39,33],[2,38,37]):get_sec([19,39,33],[20,0,16]),:16])
print(df.shape)
df.to_csv('PN00_1_plot.csv')  ## Input data for EEG-DIF