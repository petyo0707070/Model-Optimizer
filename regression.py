import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf, adfuller, arma_order_select_ic
import statsmodels.api as sm

regressor_dic = {
    'itrax_gen_10y_spread_delta' : '10Y CDS Spread Change',
    'bank_etf_return': '% change of STOXX Bank ETF',
    'german_10y_yield_delta': 'Yield change 10Y Bund'
}


stoxx_600 = pd.read_csv('stoxx_600.csv') # Load the Euro Stoxx Data
stoxx_600['datadate'] = pd.to_datetime(stoxx_600['datadate']) # Load as datetime
stoxx_600.set_index(stoxx_600['datadate'], inplace= True, drop = True) # Set as index
stoxx_600_daily = stoxx_600[['prccd']] # Keep only the price and date as that is what matters
stoxx_600_daily = stoxx_600_daily.copy()
stoxx_600_daily.loc[:,'return'] = stoxx_600_daily['prccd'].diff() / stoxx_600_daily['prccd'].shift(1) # Calculate percentage change
stoxx_600_daily.dropna(inplace = True) # drop na

spy_kbe = pd.read_csv('spy_kbe.csv') # Load the spy and kbe ETFs
spy = spy_kbe[spy_kbe['TICKER'] == 'SPY'] # Leave Only SPY
spy = spy[['date', 'PRC']] # Subset only for close and date

spy['date'] = pd.to_datetime(spy['date'], format = "%d/%m/%Y") # Load as datetime
spy.set_index(spy['date'], inplace= True, drop = True) # Set as index

spy.loc[:,'return'] = spy['PRC'].diff() / spy['PRC'].shift(1) # Calculate percentage change
spy.dropna(inplace = True) # drop na


spy_daily = spy.copy()

itrax_gen_10y_daily = pd.read_excel('test_cds.xlsx', sheet_name = 'itrax_gen_10y') # Load the Itrax Gen CDS
itrax_gen_10y_daily['spread change'] = itrax_gen_10y_daily['Last Price'].diff() # Calculate the daily spread change
itrax_gen_10y_daily.dropna(inplace= True)
itrax_gen_10y_daily['Date'] = pd.to_datetime(itrax_gen_10y_daily['Date'])
itrax_gen_10y_daily.set_index(itrax_gen_10y_daily['Date'], inplace= True, drop= True)

bank_daily = pd.read_excel('bank_stoxx.xlsx')
bank_daily['return'] = bank_daily['Last Price'].diff() / bank_daily['Last Price'].shift(1)
bank_daily.dropna(inplace= True)
bank_daily['Date'] = pd.to_datetime(bank_daily['Date'])
bank_daily.set_index(bank_daily['Date'], inplace= True, drop= True)

german_10y_daily = pd.read_excel('german_10y.xlsx')
german_10y_daily['yield change'] = german_10y_daily['Last Price'].diff()
german_10y_daily.dropna(inplace= True)
german_10y_daily['Date'] = pd.to_datetime(german_10y_daily['Date'])
german_10y_daily.set_index(german_10y_daily['Date'], inplace= True, drop= True)


# To allign all the timeseries together is to find all the common indexes
common_index = stoxx_600.index.intersection(spy_daily.index).intersection(itrax_gen_10y_daily.index).intersection(bank_daily.index).intersection(german_10y_daily.index) # Get the dates where all the timeseries have data to populate
stoxx_600_daily = stoxx_600_daily.loc[common_index]
spy_daily = spy_daily.loc[common_index]
itrax_gen_10y_daily = itrax_gen_10y_daily.loc[common_index]
bank_daily = bank_daily.loc[common_index]
german_10y_daily = german_10y_daily.loc[common_index]


# We build the final dataframe that will be used to hold all the independent variables together + the dependent variable
df_daily = pd.DataFrame(index = itrax_gen_10y_daily.index) # A dataframe to hold all the information in one
df_daily['stoxx_600_return'] = (stoxx_600_daily['return'] * 100).values # in %
df_daily['sp500_return'] = (spy_daily['return'] * 100).values # in %
df_daily['itrax_gen_10y_spread_delta'] = itrax_gen_10y_daily['spread change'].values # in bps
df_daily['bank_etf_return'] = (bank_daily['return'] * 100).values # in %
df_daily['german_10y_yield_delta'] = (german_10y_daily['yield change'] * 100).values # In bps

df_weekly = df_daily.resample('W').sum()

##### Exploratory analysis  that will be used for justifying regression choices Main Idea is to Check how Stock Market Return was influenced by CDS spreads, Banking Health Returns
# First step is to split the datasets into  pre, during and post crisis
summary_table_daily = df_daily.describe()
summary_table_weekly = df_weekly.describe()

print(summary_table_daily)
print(summary_table_weekly)

df_daily_correlation = df_daily[['itrax_gen_10y_spread_delta', 'bank_etf_return', 'german_10y_yield_delta']].corr()
sns.heatmap(df_daily_correlation, annot = True, cmap = 'coolwarm')
plt.title('Correlation Matrix of the Regressors Daily Frequency')
plt.show()

# Since our explanatory variables are quite correlated best we can do is check if there is a high degree of multicollienearity using VIF (variance inflation factor)
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data


# This checks for the presense of autocorrelation via the auto and partial auto corelation functions
def plot_acf_pacf(series, regressor, lags=10):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=axes[0])
    plt.title(f'ACF for {regressor_dic[regressor]}')
    plt.show()

    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axes[1])
    plt.title(f'PACF for {regressor_dic[regressor]}')
    plt.show()

# ADF Test for stationarity, i.e. since we are dealing with a timeseries we would like to know if the regressors themselves for stationary timeserues
def adf_test(series, regressor):
    result = adfuller(series)
    print(f'p-value for ADF test {regressor_dic[regressor]}', result[1])


# Find best lag using AIC and BIC in case of serial correlation in the regressors
def best_lag(series, regressor, max_lag=5):
    result = arma_order_select_ic(series, max_ar=max_lag, max_ma=0, ic=['aic', 'bic'])
    print(f'Best lag by AIC for: {regressor_dic[regressor]}', result.aic_min_order)
    print(f'Best lag by BIC: for {regressor_dic[regressor]}', result.bic_min_order)



vifs_daily = calculate_vif(df_daily[['itrax_gen_10y_spread_delta', 'bank_etf_return', 'german_10y_yield_delta']])
vifs_weekly = calculate_vif(df_weekly[['itrax_gen_10y_spread_delta', 'bank_etf_return', 'german_10y_yield_delta']])
print(vifs_daily)
print(vifs_weekly)

for regressor in ['itrax_gen_10y_spread_delta', 'bank_etf_return', 'german_10y_yield_delta']:
    plot_acf_pacf(df_daily[regressor], regressor)
    adf_test(df_daily[regressor], regressor)
    best_lag(df_daily[regressor], regressor)
