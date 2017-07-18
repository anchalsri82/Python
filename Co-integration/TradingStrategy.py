import pandas as pd
# Import statsmodels equivalents to validate results
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import (lagmat, add_trend)
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

# Verfify methods:
instrument1 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/C.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument1['Returns'] = np.log(instrument1['Adj Close'].astype(np.float)/instrument1['Adj Close'].shift(1).astype(np.float))
instrument1=instrument1[1:]
instrument2 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/BAC.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument2['Returns'] = np.log(instrument2['Adj Close'].astype(np.float)/instrument2['Adj Close'].shift(1).astype(np.float))
instrument2=instrument2[1:]
returns1 = instrument1['Returns'].values
returns2 = instrument2['Returns'].values
Y1_t = instrument1['Adj Close'].values
Y2_t = instrument2['Adj Close'].values

dY1_t = pd.Series(Y1_t, name='Y1_t').diff().dropna()
dY2_t = pd.Series(Y2_t, name='Y2_t').diff().dropna()

# Verfify methods:
instrument1 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/C.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument1['Returns'] = np.log(instrument1['Adj Close'].astype(np.float)/instrument1['Adj Close'].shift(1).astype(np.float))
instrument1=instrument1[1:]
instrument2 = pd.read_csv("D:/Projects/Python/trunk/Co-integration/MarketData/BAC.csv", index_col=0, parse_dates=True, dayfirst=True)
instrument2['Returns'] = np.log(instrument2['Adj Close'].astype(np.float)/instrument2['Adj Close'].shift(1).astype(np.float))
instrument2=instrument2[1:]
returns1 = instrument1['Returns'].values
returns2 = instrument2['Returns'].values
Y1_t = instrument1['Adj Close'].values
Y2_t = instrument2['Adj Close'].values

dY1_t = pd.Series(Y1_t, name='Y1_t').diff().dropna()
dY2_t = pd.Series(Y2_t, name='Y2_t').diff().dropna()

## ADF    
data = pd.concat([instrument1['Returns'], instrument2['Returns']], axis=1, keys=['Returns1', 'Returns2']) 

Yt = np.vstack((Y1_t, Y2_t))
Yr = np.vstack((returns1, returns2))
dY = np.vstack((dY1_t, dY2_t))
maxlags = int(round(12*(len(Yr)/100.)**(1/4.)))

## Optimal Legs
maxlagOptimumVectorAR = GetOptimalLag(Yr, maxlags,  modelType='VectorAR')
#maxlagOptimumADFuller = GetOptimalLag(Y=Yr, maxlags=maxlags,  modelType='ADFuller')
#resultADFuller = GetADFuller(Y=Yr, maxlags= maxlagOptimumVectorAR['bestlagaic'])

model = VAR(data)
results = model.fit(0, method='ols', ic='aic', trend='c',verbose=True)
results.summary()

#resultADFullerYt = GetADFuller(Yt, 1)
#resultADFullerYt['adfstat']

#maxlagOptimumADFullerdY = GetOptimalLag(dY, maxlags,  modelType='ADFuller')
#resultADFullerdY = GetADFuller(dY, maxlagOptimumADFullerdY['bestlagaic'])

# ENGLE-GRANGER STEP 1
Y2_t_d = np.vstack((np.ones(len(Y2_t)), Y2_t))
resultGetOLS = GetOLS(Y=Y1_t, X=Y2_t_d)

a_hat = resultGetOLS['beta_hat'][0,0]
beta2_hat = resultGetOLS['beta_hat'][0,1]

et_hat = Y1_t - np.dot(beta2_hat, Y2_t) - a_hat

# ENGLE-GRANGER STEP 2
result_et_hat_adf = GetADFuller(Y=et_hat, maxlags=1, regression='nonconstant')
print("ADF stat : %f" % result_et_hat_adf['adfstat'])


sm_result_et_hat_adf = adfuller(et_hat, maxlag=1, regression='nc', autolag=None, regresults=True)


# ===== SPREAD PLOTS  =====

from matplotlib import gridspec

plt.figure(1, figsize=(15, 20))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5])

# === SPREAD TIME SERIES ===

plt.subplot(gs[0])
plt.title('Cointegrating spread $\hat{e}_t$ (Brent & Gasoil)')
e_t_hat.plot()
plt.axhline(e_t_hat.mean(), color='red', linestyle='--') # Add the mean
plt.legend(['$\hat{e}_t$', 'mean={0:0.2g}'.format(e_t_hat.mean())], loc='lower right')
plt.xlabel('')

# === SPREAD HISTOGRAM ===

plt.subplot(gs[1])

from scipy import stats

ax = sns.distplot(e_t_hat, bins=20, kde=False, fit=stats.norm);
plt.title('Distribution of Cointegrating Spread for Brent and Gasoil')

# Get the fitted parameters used by sns
(mu, sigma) = stats.norm.fit(e_t_hat)
print "mu={0}, sigma={1}".format(mu, sigma)

# Legend and labels 
plt.legend(["normal dist. fit ($\mu \sim${0}, $\sigma=${1:.2f})".format(0, sigma),
            "$\hat{e}_t$"
           ])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('')

# # Cross-check this is indeed the case - should be overlaid over black curve
# x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
# ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
# plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),
#            "cross-check"])

# === SPREAD PACF ===

from statsmodels.graphics.tsaplots import plot_pacf

ax = plt.subplot(gs[2])
plot_pacf(e_t_hat, lags=50, alpha=0.01, ax=ax)
plt.title('')
plt.xlabel('Lags')
plt.ylabel('PACF')
# plt.text(x=40.5, y=0.85, s='PACF', size='xx-large')


# ===== GOODNESS OF FIT OF SPREAD TO NORM  ======

# Note: do not use scipy's kstest, 
# see http://stackoverflow.com/questions/7903977/implementing-a-kolmogorov-smirnov-test-in-python-scipy

import statsmodels.api as sm

# Lilliefors test http://en.wikipedia.org/wiki/Lilliefors_test
print 'Lilliefors:', sm.stats.lillifors(e_t_hat)

# Most Monte Carlo studies show that the Anderson-Darling test is more powerful 
# than the Kolmogorov-Smirnov test. It is available in scipy.stats with critical values,
# and in statsmodels with approximate p-values:

print 'Anderson-Darling:', sm.stats.normal_ad(e_t_hat)


# ===== STEP 2: ADF TEST ON SPREAD  =====

# Test spread for stationarity with ADF assuming maxlag=1, no constant and no trend
my_res_adf = my_adfuller(e_t_hat, maxlag=3, regression='nc')

# Validate result with statsmodels equivalent
sm_res_adf = adfuller(e_t_hat, maxlag=3, regression='nc', autolag=None, regresults=True)

print sm_res_adf
print my_res_adf['adfstat']
print "%0.4f" % my_res_adf['adfstat']

# ===== STABILITY CHECK  =====
print key, np.abs(my_res_adf['roots'])
print "passes stability check: {0}".format(is_stable(my_res_adf['roots']))

from statsmodels.regression.linear_model import OLS

Y = y.diff()[1:]  # must remove first element from array which is nan
X = pd.concat([x.diff()[1:], e_t_hat.shift(1)[1:]], axis=1)
X_c = add_constant(X)

sm_res_ecm = OLS(Y, X).fit()  # fit without constant
sm_res_ecm_c = OLS(Y, X_c).fit()  # fit without constant

sm_res_ecm_c.summary2()
sm_res_ecm.summary2()

# ======  FIT TO OU PROCESS  ======

# My implementations
from analysis import my_AR  # AR(p) model

# Import statsmodels equivalents to validate results
from statsmodels.tsa.ar_model import AR

# Run AR(1) model with constant term with e_t_hat as endogenous variable
my_res_ar = my_AR(endog=e_t_hat, maxlag=1, trend='c')
sm_res_ar = AR(np.array(e_t_hat)).fit(maxlag=3, trend='c', method='cmle')

# Stability Check
print 'is AR({0}) model stable: {1}'.format(sm_res_ar.k_ar, is_stable(sm_res_ar.roots))
print 'is AR({0}) model stable: {1}'.format(my_res_ar['maxlag'], is_stable(my_res_ar['roots']))

# Cross-checks
print "\
AR({12}).fit.params={0} \n MY_AR({13}) params={1} \n\
AR({12}).fit.llf={2} \n MY_AR({13}) llf={3} \n\
AR({12}).fit.nobs={4} \n MY_AR({13}) nobs={5} \n\
AR({12}).fit.cov_params(scale=ols_scale)={6} \n MY_AR({13}) cov_params={7} \n\
AR({12}).fit.bse={8} \n MY_AR({13}) bse={9} \n\
AR({12}).fit.tvalues={10} \n MY_AR({13}) tvalue={11} \n\
AR({12}).fit.k_ar={12} \n MY_AR({13}) maxlag={13} \n\
".format(
    sm_res_ar.params, my_res_ar['params'],
    sm_res_ar.llf, np.array(my_res_ar['llf']),
    sm_res_ar.nobs, my_res_ar['nobs'],
    sm_res_ar.cov_params(scale=my_res_ar['ols_scale']), my_res_ar['cov_params'],
    sm_res_ar.bse, my_res_ar['bse'],
    sm_res_ar.tvalues, my_res_ar['tvalue'],
    sm_res_ar.k_ar, my_res_ar['maxlag']
)

tau = 1. / 252.  # ok for daily frequency data

# AR(1)
my_C = my_res_ar['params'][0]
my_B = my_res_ar['params'][1]
my_theta = - np.log(my_B) / tau
my_mu_e = my_C / (1. - my_B)
my_sigma_ou = np.sqrt((2 * my_theta / (1 - np.exp(-2 * my_theta * tau))) * my_res_ar['sigma'])
my_sigma_e = my_sigma_ou / np.sqrt(2 * my_theta)
my_halflife = np.log(2) / my_theta
print ' AR({8}): my_C={0}, my_B={1}, tau={2}, my_theta={3}, my_mu_e={4}, my_sigma_ou={5}, my_sigma_e={6}, my_halflife={7:.4f}'.format(my_C, 
                                        my_B, 
                                        tau,  
                                        my_theta, 
                                        my_mu_e,
                                        my_sigma_ou,
                                        my_sigma_e,
                                        my_halflife,
                                        my_res_ar['maxlag'])

# AR(3)
sm_C = sm_res_ar.params[0]
sm_B = sm_res_ar.params[1]
sm_theta = - np.log(sm_B) / tau
sm_mu_e = sm_C / (1. - sm_B)
sm_sigma_ou = np.sqrt((2 * sm_theta / (1 - np.exp(-2 * sm_theta * tau))) * sm_res_ar.sigma2)
sm_sigma_e = sm_sigma_ou / np.sqrt(2 * sm_theta)
sm_halflife = np.log(2) / sm_theta
print ' AR({8}): sm_C={0}, sm_B={1}, tau={2}, sm_theta={3}, sm_mu_e={4}, sm_sigma_ou={5}, sm_sigma_e={6}, sm_halflife={7:.4f}'.format(
    sm_C,
    sm_B,
    tau,
    sm_theta,
    sm_mu_e,
    sm_sigma_ou,
    sm_sigma_e,
    sm_halflife,
    sm_res_ar.k_ar)

# Equivalent to AR(1) model above but expressed using differences

from statsmodels.regression.linear_model import OLS

Y_e = e_t_hat.diff()[1:]  # de_t
X_e = e_t_hat.shift(1)[1:]  # e_t-1
X_e = add_constant(X_e)
r = OLS(Y_e, X_e).fit()
X_e = X_e.iloc[:, 1]  # remove the constant now that we're done

r.summary2()

# PLOTTING

e_t_hat.plot(label='$\hat{e}_t$', figsize=(15, 7))
# plt.plot(e_t_hat, label='$\hat{e}_t$')

# Trading bounds
plt.title('Trading bounds for Cointegrated Spread (Brent & Gasoil)')
plt.axhline(0, color='grey', linestyle='-')  # axis line

plt.axhline(my_mu_e, color='green', linestyle=':', label='AR(1) OU $\mu_e \pm \sigma_{eq}$')
plt.axhline(sm_mu_e, color='red', linestyle='--', label='AR(3) OU $\mu_e \pm \sigma_{eq}$')

plt.axhline(my_sigma_e, color='green', linestyle=':')
plt.axhline(-my_sigma_e, color='green', linestyle=':')
plt.axhline(sm_sigma_e, color='red', linestyle='--')
plt.axhline(-sm_sigma_e, color='red', linestyle='--')

plt.legend(loc='lower right')

# ======== BETA HEDGING ========

# === In-sample fit ===
from statsmodels.regression.linear_model import OLS

Y_is = df_is['gasoil'].diff()[1:]  # must remove first element from array which is nan
X_is = df_is['brent'].diff()[1:]

X_is_c = add_constant(X_is)

# res_bh = OLS(Y_is, X_is).fit()  # fit without constant
res_bh_c = OLS(Y_is, X_is_c).fit()  # fit without constant

res_bh_c.summary2()


# === In-sample testing ===
print (Y_is - res_bh_c.params[1]*X_is).mean()  # long gasoil, short brent
print (Y_is).mean()  # long gasoil only


# === Out-of-sample testing ===
Y_os = df_os['gasoil'].diff()[1:]
X_os = df_os['brent'].diff()[1:]

print (Y_os - res_bh_c.params[1]*X_os).mean()  # long gasoil, short brent
print (Y_os).mean()  # long gasoil only

        
normalised_e_t_hat = static_zscore(e_t_hat, mean=sm_mu_e, sigma=sm_sigma_e)
normalised_e_t_hat.plot()
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['z-score $\hat{e}_t$', '+1', '-1'], loc='lower right')
plt.axhline(0, color='grey')


# ==== GENERATE P&L DATAFRAME FOR A GIVEN SPREAD

def get_pnl_df(spread, mean, sigma):
    """
    Note the input spread must be zscore-normalised
    """
    spread_norm = (spread - mean) / sigma  # normalise as z-score
    df_pnl_is = pd.DataFrame(index=spread.index)
    df_pnl_is['e_t_hat'] = spread
    df_pnl_is['e_t_hat_norm'] = spread_norm
    # df_pnl_is['diff'] = df_pnl_is['e_t_hat'].diff()
    df_pnl_is['pos'] = np.nan
    # Go long the spread when it is below -1 as expectation is it will rise
    df_pnl_is.loc[df_pnl_is['e_t_hat_norm'] <= -1.0, 'pos'] = 1
    # Go short the spread when it is above +1 as expectation is it will fall
    df_pnl_is.loc[df_pnl_is['e_t_hat_norm'] >= 1.0, 'pos'] = -1
    # Exit positions when close to zero
    df_pnl_is.loc[(df_pnl_is['e_t_hat_norm'] < 0.1) & (df_pnl_is['e_t_hat_norm'] > -0.1), 'pos'] = 0
    # # forward fill NaN's with previous value
    df_pnl_is['pos'] = df_pnl_is['pos'].fillna(method='pad')

    # Returns must be calculated in unnormalised spread
    df_pnl_is['chg'] = df_pnl_is['e_t_hat'].diff().shift(-1)  # adopting Boris convention with shift(-1) (must shift after taking diff)
    # PnL
    df_pnl_is['pnl'] = df_pnl_is['pos'] * df_pnl_is['chg']
    df_pnl_is['pnl_cum'] = df_pnl_is['pnl'].cumsum()
    
    return df_pnl_is

df_pnl_is = get_pnl_df(e_t_hat, mean=sm_mu_e, sigma=sm_sigma_e)
df_pnl_is.tail()

df_pnl_is.loc[df_pnl_is['pnl'].isnull(), 'pnl']

%run my_pyfolio.py

plot_drawdown_periods(df_pnl_is['pnl'], top=5)



# ======== OUT-OF-SAMPLE TESTING ========

# Construct the out-of-sample spread
e_t_hat_os = df_os['gasoil'] - c_hat - beta2_hat*df_os['brent']

# Normalise to OU bounds
normalised_e_t_hat_os = static_zscore(e_t_hat_os, mean=sm_mu_e, sigma=sm_sigma_e)
normalised_e_t_hat_os.plot()
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['z-score $\hat{e}_t$', '+1', '-1'], loc='upper right')
plt.axhline(0, color='grey')

df_pnl_os = get_pnl_df(e_t_hat_os, mean=sm_mu_e, sigma=sm_sigma_e)
df_pnl_os.tail()

df_pnl_is[:-1]['pnl_cum'][-1]

%run my_pyfolio.py
df_pnl_is.index[-1]

df_temp = df_pnl_is[:-1]['pnl_cum']  # remove nan on last row
k = df_temp[-1]  # last non-nan row of in-sample pnl
df_temp.plot()
plot_drawdown_periods(df_pnl_os['pnl'], k=k, top=5)
plt.axvline(df_temp.index[-1], color='black', linestyle='--')
plt.legend(['in-sample', 'out-of-sample', 'boundary'], loc='upper left')

