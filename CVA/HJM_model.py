from __future__ import division
import numpy as np
import pandas as pd
import math
from math import exp, sqrt, log
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed
import sys

M = 451
I = 500 
n_tau = 50 
dt = 0.01
shape_3D = (M + 1, I, n_tau + 1)
np.random.seed(1000)

# Take drift, volatilities and f(t=0, T)  from CQF 'my HJM Model - MC - Caplet v2.xlsm'
data = pd.read_csv("CVAParams.csv", index_col=0)
mu = np.array(data.iloc[0, :], dtype=np.float)
vol1 = np.array(data.iloc[1, :], dtype=np.float)
vol2 = np.array(data.iloc[2, :], dtype=np.float)
vol3 = np.array(data.iloc[3, :], dtype=np.float)
S0 = np.array(data.iloc[4, :], dtype=np.float)
tau = np.array(data.columns, dtype=np.float)
d_S_plus = np.diff(S0)
d_S_plus = np.append(d_S_plus, d_S_plus[-1])
d_tau = np.diff(tau)
d_tau = np.append(d_tau, d_tau[-1])

t_m = np.zeros((M + 1), dtype=np.float)
S_plus_m = np.zeros(shape_3D, dtype=np.float)

t_m[0] = 0.0
S_plus_m[0][:] = S0
S_minus_m = S_plus_m.copy()  

for i in xrange(1, M + 1, 1):
    if i % 10 == 0:
        print 'i=', i

    rand1 = np.random.standard_normal((I, 1))
    rand2 = np.random.standard_normal((I, 1))
    rand3 = np.random.standard_normal((I, 1))

    d_S_plus = np.diff(S_plus_m[i-1][:])
    d_S_plus = np.hstack((d_S_plus, d_S_plus[:, -1].reshape(I, 1)))
    d_S_minus = np.diff(S_minus_m[i-1][:])
    d_S_minus = np.hstack((d_S_minus, d_S_minus[:, -1].reshape(I, 1)))

	t_m[i] = round(t_m[i - 1] + dt, 2)  
    S_plus_m[i][:] = S_plus_m[i - 1][:] + \
                     mu * dt + \
                     (vol1 * rand1 + vol2 * rand2 + vol3 * rand3) * math.sqrt(dt) + \
                     (d_S_plus / d_tau) * dt
    S_minus_m[i][:] = S_minus_m[i - 1][:] + \
                     mu * dt + \
                     (vol1 * (-rand1) + vol2 * (-rand2) + vol3 * (-rand3)) * math.sqrt(dt) + \
                     (d_S_minus / d_tau) * dt


data2 = pd.read_csv("CVAInput.csv", index_col=0)

col_names = ['Sim' + str(x) for x in xrange(1, I + 1)]
f_plus = pd.DataFrame(index=data2.index, columns=col_names, dtype=np.float) 
i = 0
for index, row in f_plus.iterrows():
    f_plus.loc[index, :] = S_plus_m[int(i), :, 1]
    i += 50

freq = 0.5
L_plus = (1.0 / freq) * (np.exp(f_plus * freq) - 1.0)


K_plus = L_plus.iloc[0, :][0]

L_plus_masked = np.ma.masked_where(L_plus < K_plus, L_plus)

ZCB_plus = 1.0 / (1 + freq * L_plus)
ZCB_plus.iloc[0, :] = 1.0 
ZCB_plus = ZCB_plus.cumprod()
ZCB_plus_mean = pd.Series(index=ZCB_plus.index, data=np.mean(ZCB_plus, axis=1))

DF = pd.DataFrame(index=ZCB_plus.index, columns=list(ZCB_plus.index))
DF.loc[0.0, :] = ZCB_plus_mean

for index, row in DF.iterrows():
    if index == 0.0:
        continue
    x = DF.loc[0.0][row.index]/DF.loc[0.0, index]
	x[x > 1] = 0
    DF.loc[index, :] = x

np.fill_diagonal(DF.values, 0)
N = 1.0
V_plus = N * freq * DF.dot(L_plus - K_plus)

E_plus = np.maximum(V_plus, 0)


E_plus2 = np.array(E_plus, dtype=np.float32)
E_plus_masked = np.ma.masked_where(E_plus2 == 0, E_plus2)


EE_plus_median = pd.Series(index=E_plus.index[:-1], data=np.ma.median(E_plus_masked[:-1], axis=1))
EE_plus_median.loc[5.0] = 0.0
EE_plus_mean = pd.Series(index=E_plus.index[:-1], data=np.ma.mean(E_plus_masked[:-1], axis=1))
EE_plus_mean.loc[5.0] = 0.0

PFE_plus = pd.Series(index=E_plus.index, data=np.percentile(E_plus_masked, q=97.5, axis=1))


index_interpol = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0']

EE_plus_median_interpol = (EE_plus_median + EE_plus_median.shift(-1)) / 2.0
EE_plus_median_interpol = EE_plus_median_interpol.iloc[:-1]
EE_plus_median_interpol.index = index_interpol

DF_interpol = DF.loc[0.0, :].copy()
DF_interpol[0.0] = 1.0
DF_interpol = (DF_interpol + DF_interpol.shift(-1))/2.0
DF_interpol = DF_interpol.iloc[:-1]
DF_interpol.index = index_interpol

PD_interpol = data2['PD'].iloc[1:]
PD_interpol.index = index_interpol

# Calculate CVA
RR = 0.4
CVA = (1 - RR) * EE_plus_median_interpol * DF_interpol * PD_interpol
CVA_total = CVA.sum()
