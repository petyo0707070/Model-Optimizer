from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas_ta as ta
import scipy
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline





def main():
    df = pd.read_csv('BTCUSDT3600.csv')
    df.dropna(inplace = True)

    df['date'] = pd.to_datetime(df['date'])

    # Calculate the Hourly, Daily and Weekly Return So far
    df['Hourly Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Daily Return'] = (df['close'] - df['close'].shift(24)) / df['close'].shift(24)
    df['Weekly Return'] = (df['close'] - df['close'].shift(168)) / df['close'].shift(168)

    # Calculate the previous #hour volatility using the appropriate type of returns
    df['Hourly Volatility 168H'] = df['Hourly Return'].rolling( window = 168).std() * 100
    df['Hawkes 168H'] = calculate_hawkes(df, 0.1, 168)
    df['Reversability 72H'] = rw_ptsr( df['close'], 72)
    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['Hour'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)

    df['VolMomentum'] = df['Hourly Volatility 168H'].rolling(24).mean() - df['Hourly Volatility 168H'].rolling(168).mean()


    df['Future Change Realized Volatility 168H'] = df['Hourly Return'].shift(-167).rolling( window = 168).std() * 100 - df['Hourly Volatility 168H']
    df.dropna(inplace = True)



    # Split into training, validation and testing
    X_train, X_validation_test, y_train, y_validation_test = train_test_split(df[['Hourly Volatility 168H', "Hawkes 168H", 'Reversability 72H', 'DayOfWeek', "Hour", "VolMomentum"]], df[['Future Change Realized Volatility 168H']], test_size = 0.3, shuffle= False)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size=0.35, shuffle= False)

    #sns.heatmap(df[X_train.columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    #plt.show()


    # Insure that there is no leakage
    X_validation = X_validation[200:]
    y_validation = y_validation[200:]

    # Insure that there is no leakage
    X_test = X_test[200:]
    y_test = y_test[200:]


    #model = grid_search_validation(X_train, X_validation, y_train, y_validation)
    #model = grid_search(X_train, X_validation, y_train, y_validation)

   # Manually saved the result of the Grid Search
    #model = xgb.XGBModel(objective='reg:pseudohubererror', booster = 'gblinear', eta = '0.1')
    model = xgb.XGBModel(objective='reg:pseudohubererror', 
                         booster = 'gbtree', 
                         eta = '0.05', 
                         n_estimators = 300, 
                         max_depth = 8, 
                         min_samples_leaf=50, 
                         subsample = 0.75, 
                         colsample_bytree = 0.75,
                         max_features = "sqrt")
    
    # Try to fit a second model Random Forest Regressor in an attempt to smooth out the noise 
    model_2 = RandomForestRegressor(
                        n_estimators = 100,
                        max_depth = 8,
                        max_features= "sqrt",
                        criterion = 'squared_error',
                        bootstrap = True,
                        min_impurity_decrease = 0
    )

    # Fit a third model - Support Vector Regressor to even more smooth the predictions
    model_3 = make_pipeline(
        RobustScaler(),
        SVR(kernel= 'rbf',
            C = 0.5,
            epsilon= 0.05))

    model.fit(X_train, y_train) # Fit the model0
    model_2.fit(X_train, y_train)
    model_3.fit(X_train, y_train)

    # Visualize the performance of the model for the first 1 week of the validation set
    predictions = model.predict(X_validation)  # Get the predictions
    predictions_2 = model_2.predict(X_validation)
    predictions_3 = model_3.predict(X_validation)

    mse = mean_squared_error(y_validation, predictions)
    mape = mean_absolute_percentage_error(y_validation, predictions) # Calculate the MAPE for the validation set
    print(f"Root MSE is {round(math.sqrt(mse),3)}, MAPE is {100 * round(mape, 3)}%")


    plt.plot(y_train[0:1000].reset_index(drop = True), label = 'Realized Change Volatility next 7 days', color = 'blue') # Plot the realized Future 7 Day Volatility
    plt.plot(pd.Series(model.predict(X_train)[0:1000] ).rolling(6).mean(), label = 'XBoost Predicted Volatility Change next 7 days MA(6)', color = 'red') # Xboost Plot the predicted future 7 day volatility
    plt.plot(pd.Series(model_2.predict(X_train)[0:1000] ).rolling(6).mean(), label = 'RF Predicted Volatility Change next 7 days MA(6)', color = 'orange') # RF Plot the predicted future 7 day volatility
    plt.plot(pd.Series(model_3.predict(X_train)[0:1000] ).rolling(6).mean(), label = 'SVR Predicted Volatility Change next 7 days MA(6)', color = 'black') # SVR Plot the predicted future 7 day volatility
    plt.plot(X_train['Hourly Volatility 168H'][0:1000].reset_index(drop = True), label = '7 Day Volatility at prediction time', color = 'green')
    plt.ylabel('Volatility')
    plt.legend()
    plt.title('First 3 Week Training')
    plt.show()


    plt.plot(y_validation[0:1000].reset_index(drop = True), label = 'Realized Change Volatility next 7 days', color = 'blue') # Plot the realized Future 7 Day Volatility
    plt.plot(pd.Series(predictions[0:1000]).rolling(6).mean(), label = 'XBoost Predicted Volatility Change next 7 days MA(6)', color = 'red') # Plot the predicted future 7 day volatility
    plt.plot(pd.Series(predictions_2[0:1000]).rolling(6).mean(), label = 'RF Predicted Volatility Change next 7 days MA(6)', color = 'orange') # RF Plot the predicted future 7 day volatility
    plt.plot(pd.Series(predictions_3[0:1000]).rolling(6).mean(), label = 'SVR Predicted Volatility Change next 7 days MA(6)', color = 'black') # SVR Plot the predicted future 7 day volatility
    plt.plot( pd.Series((predictions + predictions_2 + predictions_3)[0:1000]/3).rolling(6).mean(), label = 'Aggregated Prediction', color = 'purple' )
    plt.plot(X_validation['Hourly Volatility 168H'][0:1000].reset_index(drop = True), label = '7 Day Volatility at prediction time', color = 'green')
    plt.ylabel('Volatility')
    plt.legend()
    plt.title('First 3 Week Vallidation')
    plt.show()

    print(f"Correlation between the Xboost predicted and realized volatility is {y_validation.iloc[:, 0].corr( pd.Series(predictions, index = y_validation.index) )}")
    print(f"Correlation between the RF predicted and realized volatility is {y_validation.iloc[:, 0].corr( pd.Series(predictions_2, index = y_validation.index) )}")
    print(f"Correlation between the SVR predicted and realized volatility is {y_validation.iloc[:, 0].corr( pd.Series(predictions_3, index = y_validation.index) )}")

def grid_search( X_train, X_validation, y_train, y_validation, param_grid = None):
    from sklearn.model_selection import GridSearchCV

    if param_grid == None:
    # This here will be used to do define the parameter seach for the hyper parameter optimizer
        param_grid = {
            'eta': [ x /100000  for x in range(10000, 1, -50)],
            # 'gamma': [x for x in range(0, 100, 20)]
            #'max_depth': [x for x in range(3, 20, 1)],
            'booster': ['gbtree', 'gblinear']
        }

    model = xgb.XGBModel(objective='reg:squarederror', gamma=60)  # Might have to manually reprogramme part of the model when changing the fixed parametres

    # Call the grid search class
    grid_search = GridSearchCV(
        estimator = model,
        param_grid= param_grid,
        scoring = 'neg_root_mean_squared_error',
        cv = 10,
        n_jobs = -1,
        verbose= 2
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)
    print('Best parameters', grid_search.best_params_)
    print('Training Score', - grid_search.best_score_)


    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_validation)
    mse = math.sqrt(mean_squared_error(y_validation, y_pred) )
    print('Validation Score',mse)

    return best_model

# Perform a grid search where the result from the validation is used instead of the training
def grid_search_validation(X_train, X_validation, y_train, y_validation, param_grid = None, include_hist_scores = False):
    from sklearn.model_selection import ParameterGrid

    # We can either specify now or later the paramet gird
    if param_grid == None:
        # This here will be used to do define the parameter seach for the hyper parameter optimizer
        param_grid = {
            'eta': [ x /100000  for x in range(10000, 1, -50)],
            'gamma': [x for x in range(0, 100, 20)],
            #'max_depth': [x for x in range(3, 20, 1)],
            'booster': ['gbtree', 'gblinear']
        }

    best_score = float('inf') # keep track of the best score so far
    best_params = None # Keep track of the combination of best parameters
    best_model = None # Keep track of the best model
    results = [] # Keep track of all the scores achieved

    for params in ParameterGrid(param_grid): # This is the grid search loop

        model = xgb.XGBModel(objective='reg:squarederror',  **params) # Might have to manually reprogramme part of the model when changing the fixed parametres
        model.fit(X_train, y_train) # Fit the model on the training data
        preds = model.predict(X_validation) # Predict for the Validation Data
        score = math.sqrt(mean_squared_error(y_validation, preds)) # Extracg the score, currently thats MSE
        print(f'Currently fitting: {params}')
        results.append(score) # append the scores might be useful for visualization


        # Continiously keep track of the model with the best eval score
        if score < best_score:
            best_score = score
            best_params = params
            best_model = model

    print("Best Parameters:", best_params)
    print("Best Validation MSE:", best_score)
    print(best_model.max_depth)

    if include_hist_scores:
        plt.hist(results)
        plt.title('Histogram of the Scores of the XBoost Grid Search with validation data scoring')
        plt.show()


    return best_model

def calculate_hawkes(data, kappa, lookback): # Calculate the Hawkes Process to be used as a feature
    atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), lookback) # Calculate the atr on a rolling basis
    norm_range = ( np.log(data['high']) - np.log(data['low']) ) / atr # Calculate the normalized range for each candle
    alpha = np.exp(-kappa) # That is the decay spped
    arr = norm_range.to_numpy().flatten() # Get the normalized range into an array
    output = np.zeros(len(norm_range)) # Will be used as the output of the Hawkes Process
    output[:] = np.nan
    for i in range(1, len(norm_range)):
         # Calculate Hawkes recursively start with the first normalized value then use the previous value for hawkes * decay measure + the curret value for te normalized range
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=norm_range.index) * kappa

def ordinal_patterns(arr: np.array, d: int) -> np.array:
    assert(d >= 2)
    fac = math.factorial(d)
    d1 = d - 1
    mults = []
    for i in range(1, d):
        mult = fac / math.factorial(i + 1)
        mults.append(mult)

    # Create array to put ordinal pattern in
    ordinals = np.empty(len(arr))
    ordinals[:] = np.nan

    for i in range(d1, len(arr)):
        dat = arr[i - d1:  i+1]
        pattern_ordinal = 0
        for l in range(1, d):
            count = 0
            for r in range(l):
                if dat[d1 - l] >= dat[d1 - r]:
                   count += 1

            pattern_ordinal += count * mults[l - 1]
        ordinals[i] = int(pattern_ordinal)

    return ordinals

def perm_ts_reversibility(arr: np.array):
    # Zanin, M.; Rodríguez-González, A.; Menasalvas Ruiz, E.; Papo, D. Assessing time series reversibility through permutation
    
    # Should be fairly large array, very least ~60
    assert(len(arr) >= 10)
    rev_arr = np.flip(arr)
   
    # [2:] drops 2 nan values off start of val
    pats = ordinal_patterns(arr, 3)[2:].astype(int)
    r_pats = ordinal_patterns(rev_arr, 3)[2:].astype(int)
   
    # pdf of patterns, forward and reverse time
    n = len(arr) - 2
    p_f = np.bincount(pats, minlength=6) / n 
    p_r = np.bincount(r_pats, minlength=6) / n

    if min(np.min(p_f), np.min(p_r)) > 0.0:
        rev = scipy.special.rel_entr(p_f, p_r).sum()
    else:
        rev = np.nan
        
    return rev

def rw_ptsr(arr, lookback: int):
    # Rolling window permutation time series reversibility
    arr = arr.to_numpy()
    rev = np.zeros(len(arr))
    rev[:] = np.nan
    
    lookback_ = lookback + 2
    for i in range(lookback_, len(arr)):
        dat = arr[i - lookback_ + 1: i+1]
        rev_w = perm_ts_reversibility(dat) 

        if np.isnan(rev_w):
            rev[i] = rev[i - 1]
        else:
            rev[i] = rev_w

    return rev


if __name__ == '__main__':
    main()