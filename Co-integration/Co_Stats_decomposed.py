# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from collections import defaultdict
#http://www.statsmodels.org/dev/vector_ar.html
import statsmodels.api as sm

import statsmodels.formula.api as sma
from statsmodels.tsa.api import VAR, DynamicVAR

from statsmodels.tools.tools import rank, add_constant
import statsmodels.tsa.vector_ar.util as util

from statsmodels.tools.linalg import logdet_symm


instrument1 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/C.csv", index_col=0, parse_dates=True, dayfirst=True)

instrument1['Returns'] = np.log(instrument1['Adj Close'].astype(np.float)/instrument1['Adj Close'].shift(1).astype(np.float))
instrument1=instrument1[1:]

print(instrument1['Returns'].head())

instrument2 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/BAC.csv", index_col=0, parse_dates=True, dayfirst=True)

instrument2['Returns'] = np.log(instrument2['Adj Close'].astype(np.float)/instrument2['Adj Close'].shift(1).astype(np.float))
instrument2=instrument2[1:]
len(instrument2)

print(instrument2['Returns'].head())

returns1 = instrument1['Returns'].values[1:]
returns2 = instrument2['Returns'].values[1:]

data = pd.concat([instrument1['Returns'], instrument2['Returns']], axis=1, keys=['Returns1', 'Returns2']) 

X = data['Returns1']
Y = data['Returns2']

maxlags = 1
method='ols'
ic='aic'
trend='c'
verbose=True

lags = maxlags
endog = np.asarray(data)
neqs = endog.shape[1]

if trend not in ['c', 'ct', 'ctt', 'nc']:
    raise ValueError("trend '{}' not supported for VAR".format(trend))
    
#selections = self.select_order(maxlags=maxlags, verbose=verbose)
#if maxlags is None:
#maxlags = int(round(12*(len(endog)/100.)**(1/4.)))
maxlags = 1
ics = defaultdict(list)
for p in range(maxlags + 1):
    # exclude some periods to same amount of data used for each lag
    # order
    #result = _estimate_var(p, offset=maxlags-p)
    k_trend = 1 #'c'
    offset=0
    n_totobs = len(endog)
    nobs = n_totobs - lags - offset
    endog_temp = endog[offset:]
    exog = None
    #z = util.get_var_endog(endog, lags, trend=trend, has_constant='raise')
    nobs = len(endog_temp)
    # Ravel C order, need to put in descending order
    z = np.array([endog_temp[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
    
    # Add constant, trend, etc.
    if trend != 'nc':
        has_constant='skip'
        #z = tsa.add_trend(z, prepend=True, trend=trend, has_constant=has_constant)
        nobsZ = len(z)
        z = [np.ones(z.shape[0]), z]
        z = np.column_stack(z)
        #end
    
    #return Z
    Z_raw=z
    # the following modification of z is necessary to get the same results
    # as JMulTi for the constant-term-parameter...
    
    for i in range(k_trend):
        if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
            z[:, i] += lags
        # make the same adjustment for the quadratic term
        if (np.diff(np.sqrt(z[:, i])) == 1).all():
            z[:, i] = (np.sqrt(z[:, i]) + lags)**2
    
    y_sample = endog_temp[lags:]
    # Lutkepohl p75, about 5x faster than stated formula
    params = np.linalg.lstsq(z, y_sample)[0] #B_hat
    resid = y_sample - np.dot(z, params) # e_hat - Residual/Disturbance
    
    # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
    # process $u$
    # equivalent definition
    # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
    # Z^\prime) Y
    # Ref: Lutkepohl p.75
    # df_resid right now is T - Kp - 1, which is a suggested correction
    
    avobs = len(y_sample)
    if exog is not None:
        k_trend += exog.shape[1]
    df_resid = avobs - (neqs * lags + k_trend)
    
    sse = np.dot(resid.T, resid) # Et_hat * Et_hat'
    omega = sse / df_resid #Sigma_Hat - Estimator of the residual covariance matrix with T = Nobs
    
    sigma_u_mle =  omega * df_resid / nobs
    
    varfit = VARResults(endog_temp, z, params, omega, lags, names=['Y1', 'Y2'], trend=trend, dates=data.dates, model=self, exog=exog)
    
    result = VARResultsWrapper(varfit) 
    #return VARResultsWrapper(varfit)        
    #end
    
    for k, v in result.info_criteria.items:
        ics[k].append(v)

selected_orders = dict((k, mat(v).argmin()) for k, v in ics.items)

if verbose:
    output.print_ic_table(ics, selected_orders)

selections = selected_orders
#end
if ic not in selections:
    raise Exception("%s not recognized, must be among %s"
                    % (ic, sorted(selections)))
lags = selections[ic]
if verbose:
    print('Using %d based on %s criterion' %  (lags, ic))


k_trend = 1 #'C'
exog_names = util.make_lag_names(endog_names, lags, k_trend)
nobs = len(endog) - lags

