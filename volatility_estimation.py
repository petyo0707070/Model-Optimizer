from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
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
from sklearn.linear_model import Ridge



def main():
    df = pd.read_csv('BTCUSDT3600.csv')
    df.dropna(inplace=True)

    df['date'] = pd.to_datetime(df['date'])

    # Calculate the Hourly, Daily and Weekly Return So far
    df['Hourly Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Daily Return'] = (df['close'] - df['close'].shift(24)) / df['close'].shift(24)
    df['Weekly Return'] = (df['close'] - df['close'].shift(168)) / df['close'].shift(168)

    # Calculate the previous #hour volatility using the appropriate type of returns
    df['Hourly Volatility 168H'] = df['Hourly Return'].rolling(window=168).std() * 100
    df['Hawkes 168H'] = calculate_hawkes(df, 0.1, 168)
    df['Reversability 72H'] = rw_ptsr(df['close'], 72)
    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['Hour'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)

    df['VolMomentum'] = df['Hourly Volatility 168H'].rolling(24).mean() - df['Hourly Volatility 168H'].rolling(
        168).mean()

    df['Price-Volume Correlation'] = df['close'].rolling(168).corr(df['volume'])

    df['Future Change Realized Volatility 168H'] = df['Hourly Return'].shift(-167).rolling(window=168).std() * 100 - df[
        'Hourly Volatility 168H']
    df.dropna(inplace=True)

    # Split into training, validation and testing
    X_train, X_validation_test, y_train, y_validation_test = train_test_split(
        df[['Hourly Volatility 168H', "Hawkes 168H", 'Reversability 72H', 'DayOfWeek', "Hour", "VolMomentum"]],
        df[['Future Change Realized Volatility 168H']], test_size=0.3, shuffle=False)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size=0.35,
                                                                  shuffle=False)

    #sns.heatmap(df[X_train.columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    #plt.show()

    # Insure that there is no leakage
    X_validation = X_validation[200:]
    y_validation = y_validation[200:]

    # Insure that there is no leakage
    X_test = X_test[200:]
    y_test = y_test[200:]

    fit_ltsm_neuralnet(X_train, y_train, X_validation, y_validation)


    # model = grid_search_validation(X_train, X_validation, y_train, y_validation)
    # model = grid_search(X_train, X_validation, y_train, y_validation)

    # Manually saved the result of the Grid Search
    # model = xgb.XGBModel(objective='reg:pseudohubererror', booster = 'gblinear', eta = '0.1')
    model = xgb.XGBModel(objective='reg:pseudohubererror',
                         booster='gbtree',
                         eta='0.05',
                         n_estimators=300,
                         max_depth=8,
                         subsample=0.75,
                         colsample_bytree=0.75)


    # Try to fit a second model Random Forest Regressor in an attempt to smooth out the noise
    model_2 = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        max_features="sqrt",
        criterion='friedman_mse',
        bootstrap=True,
        min_impurity_decrease=0
    )


    # Fit a third model - Support Vector Regressor to even more smooth the predictions
    model_3 = make_pipeline(
        RobustScaler(),
        SVR(kernel='rbf',
            C=0.5,
            epsilon=0.02,
            gamma = 0.1))

    models = [model, model_2, model_3] # Save the definition of the tree models, might come in handy when we need to fit models for something like aggregated predictions

    model.fit(X_train, y_train.values.ravel())  # Fit the model0
    #get_xboost_feature_importance(model, X_train)

    model_2.fit(X_train, y_train.values.ravel())
    #get_randomforest_feature_importance(model_2, X_train)

    model_3.fit(X_train, y_train.values.ravel()) # it is way too computationally intensive to calculate feature importance for SVR so no

    # Visualize the performance of the model for the first 1 week of the validation set
    predictions = model.predict(X_validation)  # Get the predictions
    predictions_2 = model_2.predict(X_validation)
    predictions_3 = model_3.predict(X_validation)

    #aggregation_model = fit_ridge_aggregation(models, X_train, y_train)

    plt.plot(y_train[0:1000].reset_index(drop=True), label='Realized Change Volatility next 7 days',
             color='blue')  # Plot the realized Future 7 Day Volatility
    plt.plot(pd.Series(model.predict(X_train)[0:1000]).rolling(6).mean(),
             label='XBoost Predicted Volatility Change next 7 days MA(6)',
             color='red')  # Xboost Plot the predicted future 7 day volatility
    plt.plot(pd.Series(model_2.predict(X_train)[0:1000]).rolling(6).mean(),
             label='RF Predicted Volatility Change next 7 days MA(6)',
             color='orange')  # RF Plot the predicted future 7 day volatility
    plt.plot(pd.Series(model_3.predict(X_train)[0:1000]).rolling(6).mean(),
             label='SVR Predicted Volatility Change next 7 days MA(6)',
             color='black')  # SVR Plot the predicted future 7 day volatility
    plt.plot(X_train['Hourly Volatility 168H'][0:1000].reset_index(drop=True),
             label='7 Day Volatility at prediction time', color='green')
    plt.ylabel('Volatility')
    plt.legend()
    plt.title('First 3 Week Training')
    plt.show()

    plt.plot(y_validation[0:1000].reset_index(drop=True), label='Realized Change Volatility next 7 days',
             color='blue')  # Plot the realized Future 7 Day Volatility
    #plt.plot(pd.Series(predictions[0:1000]).rolling(6).mean(),
    #         label='XBoost Predicted Volatility Change next 7 days MA(6)',
    #         color='red')  # Plot the predicted future 7 day volatility
    plt.plot(pd.Series(predictions_2[0:1000]).rolling(6).mean(),
             label='RF Predicted Volatility Change next 7 days MA(6)',
             color='orange')  # RF Plot the predicted future 7 day volatility
    plt.plot(pd.Series(predictions_3[0:1000]).rolling(6).mean(),
             label='SVR Predicted Volatility Change next 7 days MA(6)',
             color='black')  # SVR Plot the predicted future 7 day volatility
    plt.plot(pd.Series((predictions + predictions_2 + predictions_3)[0:1000] / 3).rolling(6).mean(),
             label='Aggregated Prediction', color='purple')
    plt.plot(X_validation['Hourly Volatility 168H'][0:1000].reset_index(drop=True),
             label='7 Day Volatility at prediction time', color='green')



    plt.ylabel('Volatility')
    plt.legend()
    plt.title('First 3 Week Vallidation')
    plt.show()

    print(
        f"Correlation between the Xboost predicted and realized volatility is {round(y_validation.iloc[:, 0].corr(pd.Series(predictions, index=y_validation.index)),3 )}, Root MSE is {round(math.sqrt(mean_squared_error(y_validation, predictions)),3)}, MAE is {round(mean_absolute_error(y_validation, predictions),3)}")
    print(
        f"Correlation between the RF predicted and realized volatility is {round( y_validation.iloc[:, 0].corr(pd.Series(predictions_2, index=y_validation.index)),3 )}, Root MSE is {round(math.sqrt(mean_squared_error(y_validation, predictions_2)),3)}, MAE is {round(mean_absolute_error(y_validation, predictions_2),3)}")
    print(
        f"Correlation between the SVR predicted and realized volatility is {round( y_validation.iloc[:, 0].corr(pd.Series(predictions_3, index=y_validation.index)), 3)}, Root MSE is {round(math.sqrt(mean_squared_error(y_validation, predictions_3)),3)}, MAE is {round(mean_absolute_error(y_validation, predictions_3),3)}")
    print(
        f"Correlation between the Aggregated prediction and realized volatility is {round( y_validation.iloc[:, 0].corr(pd.Series((predictions + predictions_2 + predictions_3)/ 3, index=y_validation.index)), 3)}, Root MSE is {round(math.sqrt(mean_squared_error(y_validation, (predictions + predictions_2 + predictions_3)/3)),3)}, MAE is {round(mean_absolute_error(y_validation, (predictions + predictions_2 + predictions_3)/3),3)}")
    

def grid_search(X_train, X_validation, y_train, y_validation, param_grid=None):
    from sklearn.model_selection import GridSearchCV

    if param_grid == None:
        # This here will be used to do define the parameter seach for the hyper parameter optimizer
        param_grid = {
            'eta': [x / 100000 for x in range(10000, 1, -50)],
            # 'gamma': [x for x in range(0, 100, 20)]
            # 'max_depth': [x for x in range(3, 20, 1)],
            'booster': ['gbtree', 'gblinear']
        }

    model = xgb.XGBModel(objective='reg:squarederror',
                         gamma=60)  # Might have to manually reprogramme part of the model when changing the fixed parametres

    # Call the grid search class
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=10,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)
    print('Best parameters', grid_search.best_params_)
    print('Training Score', - grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_validation)
    mse = math.sqrt(mean_squared_error(y_validation, y_pred))
    print('Validation Score', mse)

    return best_model


# Perform a grid search where the result from the validation is used instead of the training
def grid_search_validation(X_train, X_validation, y_train, y_validation, param_grid=None, include_hist_scores=False):
    from sklearn.model_selection import ParameterGrid

    # We can either specify now or later the paramet gird
    if param_grid == None:
        # This here will be used to do define the parameter seach for the hyper parameter optimizer
        param_grid = {
            'eta': [x / 100000 for x in range(10000, 1, -50)],
            'gamma': [x for x in range(0, 100, 20)],
            # 'max_depth': [x for x in range(3, 20, 1)],
            'booster': ['gbtree', 'gblinear']
        }

    best_score = float('inf')  # keep track of the best score so far
    best_params = None  # Keep track of the combination of best parameters
    best_model = None  # Keep track of the best model
    results = []  # Keep track of all the scores achieved

    for params in ParameterGrid(param_grid):  # This is the grid search loop

        model = xgb.XGBModel(objective='reg:squarederror',
                             **params)  # Might have to manually reprogramme part of the model when changing the fixed parametres
        model.fit(X_train, y_train)  # Fit the model on the training data
        preds = model.predict(X_validation)  # Predict for the Validation Data
        score = math.sqrt(mean_squared_error(y_validation, preds))  # Extracg the score, currently thats MSE
        print(f'Currently fitting: {params}')
        results.append(score)  # append the scores might be useful for visualization

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


def calculate_hawkes(data, kappa, lookback):  # Calculate the Hawkes Process to be used as a feature
    atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']),
                 lookback)  # Calculate the atr on a rolling basis
    norm_range = (np.log(data['high']) - np.log(data['low'])) / atr  # Calculate the normalized range for each candle
    alpha = np.exp(-kappa)  # That is the decay spped
    arr = norm_range.to_numpy().flatten()  # Get the normalized range into an array
    output = np.zeros(len(norm_range))  # Will be used as the output of the Hawkes Process
    output[:] = np.nan
    for i in range(1, len(norm_range)):
        # Calculate Hawkes recursively start with the first normalized value then use the previous value for hawkes * decay measure + the curret value for te normalized range
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=norm_range.index) * kappa


def ordinal_patterns(arr: np.array, d: int) -> np.array:
    assert (d >= 2)
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
        dat = arr[i - d1:  i + 1]
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
    assert (len(arr) >= 10)
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
        dat = arr[i - lookback_ + 1: i + 1]
        rev_w = perm_ts_reversibility(dat)

        if np.isnan(rev_w):
            rev[i] = rev[i - 1]
        else:
            rev[i] = rev_w

    return rev

def get_xboost_feature_importance(model, X_train):
    booster = model.get_booster()
    importance_dic = booster.get_score(importance_type= 'gain')

    importance = [importance_dic[x] for x in X_train.columns]


    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, importance, color='skyblue')
    plt.xlabel('Importance (Gain)')
    plt.title('Feature Importances from XGBoost Model')
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.tight_layout()
    plt.show()

def get_randomforest_feature_importance(model, X_train):
    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, importance, color='skyblue')
    plt.title('Feature Importances from Randam Forest Model')
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.tight_layout()
    plt.show()

# Fit an aggregation of the model predictions based on a Ridge Regression where the training is done using K-fold splitting i.e. train on all folds and predict the current fold
def fit_ridge_aggregation(models, X_train, y_train):
    kf = KFold(n_splits=5, shuffle= False)
    predictions = np.zeros((X_train.shape[0], 3))

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]

        models[0].fit(X_tr, y_tr.values.ravel())
        models[1].fit(X_tr, y_tr.values.ravel())
        models[2].fit(X_tr, y_tr.values.ravel())

        predictions[val_idx, 0] = models[0].predict(X_val)
        predictions[val_idx, 1] = models[1].predict(X_val)
        predictions[val_idx, 2] = models[2].predict(X_val)

    aggregation_model = Ridge()
    aggregation_model.fit(predictions, y_train)
    return aggregation_model

def fit_ltsm_neuralnet(X_train, y_train, X_validation, y_validation):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping



    def create_sequences(X, y, window_size):
        Xs, ys = [], []

        for i in range(len(X) - window_size):
            Xs.append(X[i:i + window_size])
            ys.append(y.iloc[i + window_size])
        return np.array(Xs), np.array(ys)

    # Flatten for scaling
    n_samples, n_features = X_train.shape
    X_train = X_train.to_numpy().reshape(-1, n_features)
    X_validation = X_validation.to_numpy().reshape(-1, n_features)

    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    window_size = 24
    X_seq_train, y_seq_train = create_sequences(X_train, y_train, window_size)
    X_seq_val, y_seq_val = create_sequences(X_validation, y_validation, window_size)

    # Now get shape
    n_timesteps, n_features = X_seq_train.shape[1], X_seq_train.shape[2]

    model = Sequential([
        LSTM(64, input_shape=(n_timesteps, n_features),
                return_sequences=False),
                Dropout(0.2), Dense(1)
    ])

    model.compile(optimizer='adam', loss='huber')
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    model.fit(X_seq_train, y_seq_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop])
    loss = model.evaluate(X_seq_val, y_seq_val)
    print(f'LSTM Loss Function i.e. Huber Loss is {loss}')

    predictions = model.predict(X_seq_val)

    plt.plot(y_validation[window_size:1000 + window_size].reset_index(drop=True), label='Realized Change Volatility next 7 days', color='blue')  # Plot the realized Future 7 Day Volatility
    plt.plot(pd.Series(predictions.reshape(-1)[0:1000]).rolling(6).mean(), label='RF Predicted Volatility Change next 7 days MA(6)', color='orange')  # RF Plot the predicted future 7 day volatility
    plt.show()
    print(
        f"Correlation between the Xboost predicted and realized volatility is {round(y_validation.iloc[24:, 0].corr(pd.Series(predictions.ravel(), index = y_validation.index[24:])),3 )}, Root MSE is {round(math.sqrt(mean_squared_error(y_validation[24:], predictions.ravel())),3)}, MAE is {round(mean_absolute_error(y_validation[24:], predictions.ravel()),3)}")


if (__name__ == '__main__'):
    main()