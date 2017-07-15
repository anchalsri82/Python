# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat
#http://www.statsmodels.org/dev/vector_ar.html

def GetOLS(Y,Z,nlag):
    nobs = Z.shape[1]
    rank = Z.shape[0]
    #noofVariables = Z.shape[0]
    covariance = np.linalg.inv(np.dot(Z, Z.T))  # [(ZZ')^-1] variance-covariance factor
    beta_hat = np.dot(np.dot(Y, Z.T,), covariance) # beta_hat YZ'(ZZ')^-1
    resid_hat = Y - np.dot(beta_hat, Z) # resid_hat = Y - beta_hat*Z
    df_resid = np.float(nobs - rank)
    rri = np.dot(resid_hat, resid_hat.T)  # resid_hat*resid_hat' 
    
    sigma_hat = rri / nobs #  'sigma_hat' -Estimator of the residual covariance martrix with T = Nobs
    ols_scale = rri / df_resid # OLS estimator for cov matrix
    
    cov_params = np.kron(covariance, ols_scale)  # covariance matrix of parameters
    bvar = np.diag(cov_params)  # variances - diagonal of covariance matrix
    stderr = np.sqrt(bvar)  # standard error
    stderr = stderr.reshape((beta_hat.shape[0], beta_hat.shape[1]), order='C')
    
    tvalues = beta_hat / stderr  # t-statistic for a given parameter estimate
    nobs2 = nobs / 2.0
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(sigma_hat) - nobs2  # log-likelihood function
    df_model = rank  # degrees of freedom of model
    eigenvalues = np.roots(np.r_[1,-beta_hat[0]])  # eigen values
    roots = eigenvalues ** -1  #roots
    
    is_Stable = np.all(np.abs(roots) > 1)
    K_dash = 2 * (2*nlag + 1)
   	
    AIC = np.log(np.absolute(np.linalg.det(sigma_hat))) + 2.0 * K_dash / (nobs) # log(sigma_hat) + 2*K_dash/T        
    BIC = np.log(np.absolute(np.linalg.det(sigma_hat))) + (K_dash/ nobs) * np.log(nobs) # log(sigma_hat) + K_dash/T*log(T)
    resultOLS = {'Z': Z,
    'Y': Y,
    'beta_hat': beta_hat,
    'resid_hat': resid_hat,
    'nobs': nobs,
    'df_resid': df_resid,
    'rri': rri,
    'sigma_hat': sigma_hat,
    'ols_scale': ols_scale,
    'cov_params': cov_params,
    'stderr': stderr,
    'tvalues': tvalues,
    'llf': llf,
    'df_model': df_model,
    'roots': roots,
    'is_Stable': is_Stable,
    'AIC': AIC,
    'BIC': BIC,
    'K_dash':K_dash
    }
    return resultOLS

def GetADFuller(Y, maxlags=None):
    #noofVariables = Y.shape[0]
    nobs = Y.shape[1]
    
    Y = np.asarray(Y)  # ensure it is in array form
    dY = np.diff(Y)
    dYT = dY.T
    dYtk = lagmat(dYT[:, :None], maxlags, trim='both', original='in')  #dY_{t-k} 
    
    nobs = dYtk.shape[0]
    dYshort = dY[:,:nobs]
    
    Z = dYtk[:, :maxlags + 1] # exogenous var
    Z = Z.T
    resultADFuller = GetOLS(Y=dYshort, Z=Z, nlag=maxlags)  # do the usual regression using OLS to estimate parameters
    return resultADFuller

def GetVectorAR(Y, maxlags=None):
    nobs = Y.shape[1]
    Yshort = Y[:,maxlags:]
    Z =  np.ones(nobs-maxlags)
    if maxlags == 0:
        Z = np.ones(nobs-maxlags)[None,:]
    else:
        for j in range(1,maxlags+1):
            Z = np.vstack((Z, Y[:,maxlags-j:-j]))
    resultVectorAR = GetOLS(Y=Yshort, Z=Z, nlag=maxlags)
    return resultVectorAR
		
def GetOptimalLag(Y,maxlags, modelType='VectorAR'):
    result={}
    for nlag in range(0, maxlags+1):
        if modelType == 'VectorAR':
            result[nlag] = GetVectorAR(Y, maxlags=nlag)
        elif modelType == 'ADFuller':
            result[nlag] = GetADFuller(Y, maxlag=nlag)
    aicbest, bestlagaic = min((v['AIC'], k) for k, v in result.items())
    bicbest, bestlagbic = min((v['BIC'], k) for k, v in result.items())
    results = {'aicbest': aicbest,
    'bestlagaic': bestlagaic,
    'bicbest': bicbest,
    'bestlagbic': bestlagbic
    }
    return results
#end GetOptimalLag

# Verfify methods:
instrument1 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/C.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument1['Returns'] = np.log(instrument1['Adj Close'].astype(np.float)/instrument1['Adj Close'].shift(1).astype(np.float))
instrument1=instrument1[1:]
instrument2 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/BAC.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument2['Returns'] = np.log(instrument2['Adj Close'].astype(np.float)/instrument2['Adj Close'].shift(1).astype(np.float))
instrument2=instrument2[1:]
returns1 = instrument1['Returns'].values
returns2 = instrument2['Returns'].values

## ADF    
data = pd.concat([instrument1['Returns'], instrument2['Returns']], axis=1, keys=['Returns1', 'Returns2']) 
#adfuller(x=data['Returns1'], maxlag=1, regression='c', autolag=None, regresults=True)
#model = VAR(data)
#daresults = model.fit(1, method='ols', ic='aic', trend='c',verbose=True)
#daresults.summary()

Y = np.vstack((returns1, returns2))

maxlags = int(round(12*(len(Y)/100.)**(1/4.)))
maxlagOptimum = GetOptimalLag(Y, 7)
result = GetADFuller(Y, maxlagOptimum['bestlagaic'])

from statsmodels.tsa.api import VAR
model = VAR(data)
results = model.fit(0, method='ols', ic='aic', trend='c',verbose=True)
results.summary()