# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#http://www.statsmodels.org/dev/vector_ar.html
import statsmodels.api as sm
import statsmodels.formula.api as sma
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tools.tools import rank, add_constant
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tsa.stattools import adfuller

instrument1 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/C.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument1['Returns'] = np.log(instrument1['Adj Close'].astype(np.float)/instrument1['Adj Close'].shift(1).astype(np.float))
instrument1=instrument1[1:]
instrument2 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/BAC.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument2['Returns'] = np.log(instrument2['Adj Close'].astype(np.float)/instrument2['Adj Close'].shift(1).astype(np.float))
instrument2=instrument2[1:]
returns1 = instrument1['Returns'].values[1:]
returns2 = instrument2['Returns'].values[1:]

data = pd.concat([instrument1['Returns'], instrument2['Returns']], axis=1, keys=['Returns1', 'Returns2']) 
model = VAR(data)
results = model.fit(1, method='ols', ic='aic', trend='c',verbose=True)
results.summary()

nobs = returns1.size

for nlag in range(1,50):
    Z =  np.ones(nobs-nlag+1)
    returns1 = instrument1['Returns'].values[nlag:]
    returns2 = instrument2['Returns'].values[nlag:]
    Y = np.vstack((returns1, returns2))
    noofVariables = Y.shape[0]
    
    for j in range(1,nlag+1):
        Z = np.vstack((Z, np.vstack((instrument1['Returns'].values[nlag-j:-j], instrument2['Returns'].values[nlag-j:-j]))))
    
    nobs = Z.shape[1]
    rank = Z.shape[0]    
    
    covariance = np.linalg.inv(np.dot(Z, Z.T))  # [(ZZ')^-1] variance-covariance factor
    beta_hat = np.dot( np.dot(Y, Z.T,), covariance) # beta_hat YZ'(ZZ')^-1
    resid_hat = Y - np.dot(beta_hat, Z) # resid_hat = Y - beta_hat*Z
    df_resid = np.float(nobs - rank)
    ssr = np.dot(resid_hat, resid_hat.T)  # resid_hat*resid_hat' Estimator of the residual covariance martrix with T = Nobs
    
    sigma_hat = ssr / nobs #  'sigma_hat' or 'scale' in statsmodels (MLE estimator for cov matrix)
    ols_scale = ssr / df_resid # OLS estimator for cov matrix
    
    cov_params = np.kron(covariance, ols_scale)  # covariance matrix of parameters
    bvar = np.diag(cov_params.T)  # entries on the diagonal of the covariance matrix  are the variances
    bse = np.sqrt(bvar)  # must take square root to get standard error
    bse = bse.reshape((noofVariables, noofVariables * nlag + 1), order='C')
    
    tvalues = beta_hat / bse  # t-statistic for a given parameter estimate
    nobs2 = nobs / 2.0
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(sigma_hat) - nobs2  # log-likelihood function of OLS model
    df_model = rank  # degrees of freedom of model
    eigenvalues = np.roots(np.r_[1,-beta_hat[0]])  # prepend 1 to -ve of params array to get characteristic equation
    roots = eigenvalues ** -1  # these are used to test for stability in is_stable method
    
    is_Stable = np.all(np.abs(roots) > 1)
    K_dash =   2 * nlag+1
    AIC = np.log(np.linalg.det(sigma_hat)) + 2.0/nobs * K_dash # log(sigma_hat) + 2*K_dash/T        
    BIC = np.log(np.linalg.det(sigma_hat)) +  (K_dash/ nobs) * np.log(nobs) # log(sigma_hat) + K_dash/T*log(T)
    
    ## ADF
    tvalues    
    adfuller(x=data['Returns1'], maxlag=1, regression='c', autolag=None, regresults=True)
