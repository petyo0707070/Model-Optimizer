from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('BTCUSDT3600.csv')
    df.dropna(inplace = True)

    # Calculate the Hourly, Daily and Weekly Return So far
    df['Hourly Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Daily Return'] = (df['close'] - df['close'].shift(24)) / df['close'].shift(24)
    df['Weekly Return'] = (df['close'] - df['close'].shift(168)) / df['close'].shift(168)

    # Calculate the previous #hour volatility using the appropriate type of returns
    df['Hourly Volatility 168H'] = df['Hourly Return'].rolling( window = 168).std() * 100
    df['Daily Volatility 720H'] = df['Daily Return'].rolling( window = 720).std() * 100
    df['Weekly Volatility 1680H'] = df['Weekly Return'].rolling( window = 1680).std() * 100


    df['Future Realized Volatility 168H'] = df['Hourly Return'].shift(-167).rolling( window = 168).std() * 100
    df.dropna(inplace = True)


    # Split into training, validation and testing
    X_train, X_validation_test, y_train, y_validation_test = train_test_split(df[['Hourly Volatility 168H', 'Daily Volatility 720H', 'Weekly Volatility 1680H']], df[['Future Realized Volatility 168H']], test_size = 0.3, shuffle= False)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size=0.35, shuffle= False)

    # Insure that there is no leakage
    X_validation = X_validation[200:]
    y_validation = y_validation[200:]

    # Insure that there is no leakage
    X_test = X_test[200:]
    y_test = y_test[200:]


    #model = grid_search_validation(X_tran, X_validation, y_train, y_validation)
    #model = grid_search(X_tran, X_validation, y_train, y_validation)

   # Manually saved the result of the Grid Search
    model = xgb.XGBModel(objective='reg:squarederror', booster = 'gblinear', eta = '0.1', gamma=60)
    model.fit(X_train, y_train) # Fit the model


    # Visualize the performance of the model for the first 1 week of the validation set
    predictions = model.predict(X_validation)  # Get the predictions
    plt.plot(y_validation[:168].reset_index(drop = True), label = 'Realized Volatility next 7 days', color = 'blue') # Plot the realized Future 7 Day Volatility
    plt.plot(predictions[:168], label = 'Predicted Volatility next 7 days', color = 'red') # Plot the predicted future 7 day volatility
    plt.ylabel('Volatility')
    plt.legend()
    plt.title('First 1 Week Vallidation')
    plt.show()

def grid_search( X_train, X_validation, y_train, y_validation, param_grid = None):
    from sklearn.model_selection import GridSearchCV

    if param_grid == None:
    # This here will be used to do define the parameter seach for the hyper parameter optimizer
        param_grid = {
            'eta': [ x /100000  for x in range(10000, 1, -50)],
            # 'gamma': [x for x in range(0, 100, 20)]
            'max_depth': [x for x in range(3, 20, 1)],
            'booster': ['gbtree', 'gblinear']
        }

    model = xgb.XGBModel(objective='reg:squarederror', gamma=60)  # Might have to manually reprogramme part of the model when changing the fixed parametres

    # Call the grid search class
    grid_search = GridSearchCV(
        estimator = model,
        param_grid= param_grid,
        scoring = 'neg_mean_squared_error',
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
    mse = mean_squared_error(y_validation, y_pred)
    print('Validation Score',mse)

    return best_model

# Perform a grid search where the result from the validation is used instead of the training
def grid_search_validation(X_train, X_validation, y_train, y_validation, param_grid = None, include_hist_scores = False):
    from sklearn.model_selection import ParameterGrid

    # We can either specify now or later the paramet gird
    if param_grid == None:
        # This here will be used to do define the parameter seach for the hyper parameter optimizer
        param_grid = {
            #eta': [ x /100000  for x in range(10000, 1, -50)],
            # 'gamma': [x for x in range(0, 100, 20)]
            'max_depth': [x for x in range(3, 20, 1)],
            #'booster': ['gbtree', 'gblinear']
        }

    best_score = float('inf') # keep track of the best score so far
    best_params = None # Keep track of the combination of best parameters
    best_model = None # Keep track of the best model
    results = [] # Keep track of all the scores achieved

    for params in ParameterGrid(param_grid): # This is the grid search loop

        model = xgb.XGBModel(objective='reg:squarederror', booster = 'gbtree',eta = 0.004, gamma=60, **params) # Might have to manually reprogramme part of the model when changing the fixed parametres
        model.fit(X_train, y_train) # Fit the model on the training data
        preds = model.predict(X_validation) # Predict for the Validation Data
        score = mean_squared_error(y_validation, preds) # Extracg the score, currently thats MSE
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



if __name__ == '__main__':
    main()