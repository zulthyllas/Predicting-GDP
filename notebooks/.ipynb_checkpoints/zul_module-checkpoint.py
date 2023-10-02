# Libraries Used

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import pmdarima as pm
from pmdarima.arima import ADFTest, auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller  
from prophet import Prophet
from tbats import BATS, TBATS
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Data Inspector
#Checks dataframe and returns shape, dtypes, sum of null values, and head

def data_inspector(df):
    print(" Shape ".center(40, "="))
    print(f'No. of Rows: {df.shape[0]}')
    print(f'No. of Columns:{df.shape[1]}')
    print()
    print(" DTypes ".center(40, "="))
    print(df.dtypes)
    print()
    print(" Null Sum ".center(40, "="))
    print(df.isnull().sum())
    return df.head()

# Quick Check 
#returns shape and head

def quick_check(df):
    print(f'The shape is {df.shape}.')
    return df.head()

# Outlier Adjuster, removes rows of outliers present in a feature.

def column_outlier(df, column_name):
        
    # Calculate IQR for the specific column
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove rows with outliers in the specific column
    outliers_removed = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    print(f'Outliers within {column_name} removed.')
    
    return outliers_removed

# Column Uniqueness return number of unique values and what values they are

def unique_attributes(df, column_name):
    print(f'There are {df[column_name].nunique()} unique values in {column_name}.')
    print(f'The unique values in {column_name} is: {df[column_name].unique()}')

# Duplicate remover

def df_duplicate_remover(df):
    print(f'Function has detected {len(df[df.duplicated()])} rows of duplicated data.')
    print()
    print(f'There is {df.shape[0]} rows before removing duplicates.')
    df = df.drop_duplicates()
    print()
    print(f'There is now {df.shape[0]} rows after removing duplicates.')
    return df
    
# Duplicate remover within columns

def column_duplicate_remover(df, column_name):
    print(f'Function has detected {len(df[df.duplicated()])} rows of duplicated data.')
    print()
    print(f'There is {df.shape[0]} rows before removing duplicates.')
    df = df.drop_duplicates(subset=column_name)
    print()
    print(f'There is now {df.shape[0]} rows after removing duplicates.')
    return df

# Merge multiple dataframes (Monthly)

def merge_dataframes_m(dataframes):
    # Check if the list of dataframes is empty
    if not dataframes:
        return None

    # Create copies of the DataFrames to avoid modifying the originals
    dataframes_copy = [df.copy() for df in dataframes]

    # Initialize the merged DataFrame with the first DataFrame copy
    merged_df = dataframes_copy[0]

    # Set 'Period' as the index for the first DataFrame copy
    merged_df.set_index('Period', inplace=True)

    # Loop through the remaining DataFrame copies and merge them
    for df in dataframes_copy[1:]:
        # Set 'Period' as the index for the current DataFrame copy
        df.set_index('Period', inplace=True)

        # Join the current DataFrame copy with the merged DataFrame on the index
        merged_df = merged_df.join(df, how='inner')

    return merged_df

# Merge multiple dataframes (Quarterly)
    
def merge_dataframes_q(dataframes):
    # Check if the list of dataframes is empty
    if not dataframes:
        return None

    # Create copies of the DataFrames to avoid modifying the originals
    dataframes_copy = [df.copy() for df in dataframes]

    # Initialize the merged DataFrame with the first DataFrame copy
    merged_df = dataframes_copy[0]

    # Set 'Period' as the index for the first DataFrame copy
    merged_df.set_index('Period', inplace=True)

    # Loop through the remaining DataFrame copies and merge them
    for df in dataframes_copy[1:]:
        # Set 'Period' as the index for the current DataFrame copy
        df.set_index('Period', inplace=True)

        # Join the current DataFrame copy with the merged DataFrame on the index
        merged_df = merged_df.join(df, how='inner')

    # Resample the merged DataFrame to quarterly frequency (Q)
    merged_df = merged_df.resample('Q').sum()

    return merged_df
    
# Time Series Line Plotter

def plot_series(df, col, title='Title', xlab=None, ylab=None, steps=1):
    
    # Set figure size to be (18, 9).
    plt.figure(figsize=(18,9))
               
    # Generate a line plot of the column name.
    # You only have to specify what goes on y-axis, since our index will auto be taken as the datetime index in matplotlib &  represented on x-axis.
    plt.plot(df[col], alpha = 0.5)
        
    # Generate title and labels.
    plt.title(title, fontsize=14)
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    
    # Enlarge tick marks.
    plt.yticks(fontsize=12)
    plt.xticks(df.index[0::steps], fontsize=12);

# Distributions & Correlations plotter

def distributions_correlations(df):
    # Histogram
    df.hist(bins=20, edgecolor='black', figsize=(12, 12));

    # Correlation heatmap
    plt.figure(figsize=(12, 12))

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['number'])

    # Compute the correlation matrix
    corr_matrix = numeric_columns.corr()

    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, linewidths=0.5, cmap='coolwarm', mask=np.triu(np.ones(corr_matrix.shape), k=1));
    # Subplots for all features of a dataframe

    # Pairplot
    sns.pairplot(df, corner = True);

def lineplots_for_features(df):
    # Get the list of column names (features)
    features = df.columns

    # Calculate the number of rows and columns for subplots
    num_features = len(features)
    num_rows = math.ceil(num_features / 3)  # Assuming 3 subplots per row
    num_cols = 3

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    # Flatten the axes array if there's only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Create a subplot for the current feature
        axes[row_idx, col_idx].plot(df[feature])
        axes[row_idx, col_idx].set_title(feature)
        
    # Adjust spacing
    plt.tight_layout()

    # Show the subplots
    plt.show()

# Plot all features in same chart

def plot_all_features(df):

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Index')
    plt.ylabel('Feature Values')
    plt.legend()
    plt.title('Line Plot of All Features')
    plt.grid(True)

    plt.show()

# Carry out statistical T test between 2 features of a dataframe

def compare_columns(df, column1_name, column2_name, alpha=0.05):
    """
    Compare the means of two columns in a DataFrame using a t-test.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column1_name (str): Name of the first column to compare.
        column2_name (str): Name of the second column to compare.
        alpha (float): The significance level (default is 0.05 for 95% confidence).

    Returns:
        str: A result message indicating the statistical similarity.
    """
    column1 = df[column1_name]
    column2 = df[column2_name]

    result = stats.ttest_ind(column1, column2)

    p_value = result.pvalue

    print('Null Hypothesis: The features specified are NOT statistically different')
    print()
    print(f'P Value: {p_value}')
    print(f'Alpha: {alpha}')
    
    if p_value < alpha:
        return "The two columns are statistically different."
    else:
        return "There is no significant difference between the two columns."
        
    # If the p-value is less than alpha, you can conclude that there is a statistically significant difference between the two columns. 
    # Otherwise, you can conclude that there is no significant difference.
        
# ADF loop for all features in dataframe

def df_loop(df, alpha):
    for feature in df.columns:
        result = df_test(alpha, df[feature])
        print(f"Feature: {feature}, P-value: {result}")

# ADF test for specific column in dataframe

def df_test(alpha, target_column):
    adf_test = ADFTest(alpha = alpha)
    return adf_test.should_diff(target_column)

# Plots ACF for all features of a dataframe

def plot_acf_for_features(df, dataframe_name, lags=20):
    # Iterate through all columns (features) in the DataFrame
    for feature in df.columns:
        # Plot ACF for the current feature
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(df[feature], lags=lags, ax=ax)
        plt.title(f'ACF for {dataframe_name} - {feature}')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()

    
# Rolling Mean Plotter

def rolling_mean(df, column_name, rm1, rm2, rm3, rm1label, rm2label, rm3label, steps = 1, title = 'Title', xlab=None, ylab=None):
    plt.figure(figsize=(18,9))
    plt.plot(df[column_name], label='Weekly') # plot y using x as index
    plt.plot(df[column_name].rolling(rm1).mean(), label=rm1label)
    plt.plot(df[column_name].rolling(rm2).mean(), label=rm2label)
    plt.plot(df[column_name].rolling(rm3).mean(), label=rm3label)
    plt.legend()
    
    # Generate title and labels.
    plt.title(title, fontsize=14)
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    
    # Enlarge tick marks.
    plt.yticks(fontsize=12)
    plt.xticks(df.index[0::steps], fontsize=12);


# Rolling Mean Plotter Without default

def rolling_mean_only(df, column_name, rm1, rm2, rm3, rm1label, rm2label, rm3label, steps = 1, title = 'Title', xlab=None, ylab=None):
    plt.figure(figsize=(18,9))
    plt.plot(df[column_name].rolling(rm1).mean(), label=rm1label, alpha=0.3)
    plt.plot(df[column_name].rolling(rm2).mean(), label=rm2label, alpha=0.5)
    plt.plot(df[column_name].rolling(rm3).mean(), label=rm3label, alpha=0.7)
    plt.legend()
    
    # Generate title and labels.
    plt.title(title, fontsize=14)
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    
    # Enlarge tick marks.
    plt.yticks(fontsize=12)
    plt.xticks(df.index[0::steps], fontsize=12);

# Autocorrelator for list of lags

def autocorrelator(series, list_of_lags):
    for i in list_of_lags:
        autocorr = series.autocorr(lag = i)
        print(f'Autocorrelation at lag {i}: {autocorr:.4f}')


# Sequence check to check for sequence of dates between train and test.

def sequence_check(y_train, y_test):
    print('Each split data head and tail will be printed in sequence of train head, train tail, test head, test tail')
    print(y_train.head())
    print(y_train.tail())
    print(y_test.head())
    print(y_test.tail())

# Pyramid ARIMA Pipeline

def arima_pipeline(df, y_train, y_test, sp=0, mp=10, d=None, sq=0, mq=10, steps=52, criterion='aic'):

    print('Instantiating model and printing grid search values:')
    print()

    # Instantiate Model with specified criterion
    arima_model = pm.AutoARIMA(
        start_p=sp,
        max_p=mp,
        d=d,
        start_q=sq,
        max_q=mq,
        trace=True,
        random_state=123,
        n_fits=50,
        information_criterion=criterion  # Choose AIC or BIC
    )

    # Fit to train
    arima_model.fit(y_train)
    print()

    # Print Model Summary
    print(arima_model.summary())
    print()

    y_pred_train = arima_model.predict(n_periods=len(y_train))
    y_pred = arima_model.predict(n_periods=len(y_test)) # we want as many future predictions as there are in y_test

    # Calculate RMSE
    print('========== RMSE ==========')
    print()
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'ARIMA Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'ARIMA Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()

    # Plot predictions to compare vs y_test.
    plt.figure(figsize=(30, 15))

    # Plot training data
    plt.plot(y_train.index, y_train, color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='ARIMA predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(df.index[0::steps], fontsize=16)

    return y_pred

# Show predicted values and test values

def show_values(y_pred, y_test):
    print('========== Predicted Values ==========')
    print(y_pred)
    print('========== Test Values ==========')
    print(y_test)


# SARIMA Pipeline

def sarima_pipeline(df, y_train, y_test, sp=0, mp=10, d=None, sq=0, mq=10, sP=0, mP=0, D=None, sQ=0, mQ=0, m=52, steps=24):
    print('Instantiating model and printing grid search values:')
    print()
    # Instantiate Model
    sarima_model = pm.AutoARIMA(
                           # same parameters as Auto ARIMA
                           start_p=sp, max_p=mp, # we specify the start and end for our grid search
                           d=d,    # let Auto ARIMA automatically find optimum value of d automatically
                           start_q=sq, max_q=mq, # we specify the start and end for our grid search
                           
                           # uncomment these hyperparameter declarations to train SARIMA model this time
                           start_P=sP, max_P=mP, # tune `P` SARIMA hyperparameter between 0 to 10 same as `p`
                           D=D,    # let Auto ARIMA automatically find optimum value of D automatically
                           start_Q=sQ, max_Q=mQ, # tune `Q` SARIMA hyperparameter between 0 to 10 same as `q`
                           m=m, # this is the `S`! Since the peak occurs every 3 months!
                           seasonal=True, # HAS to be set to True to use `m`. goes in conjunction with `m`
                           
                           # same parameters as Auto ARIMA
                           trace=True, # Print values for each fit in the grid search
                           random_state=123,
                           n_fits=50
                          )
    # Fit to train
    sarima_model.fit(y_train)
    print()

    # Print Model Summary
    print(sarima_model.summary())
    print()

    y_pred_train = sarima_model.predict(n_periods=len(y_train))
    y_pred = sarima_model.predict(n_periods=len(y_test)) # we want as many future predictions as there are in y_test

     # Calculate RMSE
    print('========== RMSE ==========')
    print()
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'SARIMA Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'SARIMA Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()
    
    # Plot predictions to compare vs y_test.
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(y_train.index, y_train, color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='SARIMA predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(df.index[0::steps], fontsize=16);
    
    return y_pred

# SARIMAX Pipeline

def sarimax_pipeline(df, X_train, y_train, X_test, y_test, sp=0, mp=10, d=None, sq=0, mq=10, sP=0, mP=0, D=None, sQ=0, mQ=0, m=52, steps=24):
    print('Instantiating model and printing grid search values:')
    print()
    # Instantiate Model
    sarima_model = pm.AutoARIMA(
                           # same parameters as Auto ARIMA
                           start_p=sp, max_p=mp, # we specify the start and end for our grid search
                           d=d,    # let Auto ARIMA automatically find optimum value of d automatically
                           start_q=sq, max_q=mq, # we specify the start and end for our grid search
                           
                           # uncomment these hyperparameter declarations to train SARIMA model this time
                           start_P=sP, max_P=mP, # tune `P` SARIMA hyperparameter between 0 to 10 same as `p`
                           D=D,    # let Auto ARIMA automatically find optimum value of D automatically
                           start_Q=sQ, max_Q=mQ, # tune `Q` SARIMA hyperparameter between 0 to 10 same as `q`
                           m=m, # this is the `S`! Since the peak occurs every 3 months!
                           seasonal=True, # HAS to be set to True to use `m`. goes in conjunction with `m`
                           
                           # same parameters as Auto ARIMA
                           trace=True, # Print values for each fit in the grid search
                           random_state=123,
                           n_fits=50
                          )
    # Fit to train
    sarima_model.fit(y_train, X_train)
    print()

    # Print Model Summary
    print(sarima_model.summary())
    print()

    y_pred_train = sarima_model.predict(n_periods=len(y_train), X = X_train)
    y_pred = sarima_model.predict(n_periods=len(y_test), X = X_test) # we want as many future predictions as there are in y_test

     # Calculate RMSE
    print('========== RMSE ==========')
    print()
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'SARIMAX Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'SARIMAX Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()
    
    # Plot predictions to compare vs y_test.
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(y_train.index, y_train, color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='SARIMAX predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(df.index[0::steps], fontsize=16);
    
    return y_pred

# Prophet Pipeline

def prophet_pipeline(df, target_df, periods=10, freq='Q', target='target', steps=24):

    # Removing rows of df to be 'predicted'
    time_series = df[:-periods]
        
    # Instantiating
    m = Prophet()
        
    # Fit to data
    m.fit(time_series)

    # Predict 'future'
    future = m.make_future_dataframe(
    periods=periods,
    freq=freq,
    )

    # Define df variable to contain predictions
    forecast = m.predict(future)

    # Plotting prophet specific plots
    print('Plotting Prophet specific forecast plot and forecast components plot.')
    print(f'y: {target}, ds: Period')
    m.plot(forecast)
    m.plot_components(forecast)

    # Re-indexing ds as index for df 
    forecast = forecast.set_index('ds')

    # Include target variable in forecast df for other studies by creating new column with original target
    forecast[target] = target_df[target]

    # Calculating RMSE 
    y_train = forecast[target][:-periods]
    y_pred_train = forecast['yhat'][:-periods]

    y_test = forecast[target][-periods:]
    y_pred = forecast['yhat'][-periods:]
    
    print('========== RMSE ==========')
    print()
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'Prophet Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Prophet Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()

    # Plot predictions compared vs y_test
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(y_train.index, y_train, color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='Prophet predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(target_df.index[0::steps], fontsize=16);

    print('========== Predicted Value(s) ==========')
    print(y_pred)
    print('========== Test Value(s) ==========')
    print(y_test)
    
    # Return forecast
    return forecast
        
# BATS Pipeline

def bats(df, target, period, steps):

    # Instantiate
    estimator = BATS()

    # Fit
    fitted_model = estimator.fit(df[target][:-period])

    # Forecast
    y_forecasted = fitted_model.forecast(steps=period)

    # Create Predicted DF
    bats_pred = df[[target]][-period:]
    bats_pred['yhat'] = y_forecasted

    # Calculating RMSE 
    y_train = df[target][:-period]
    #y_pred_train = forecast['yhat'][:-periods]

    y_test = bats_pred[target]
    y_pred = bats_pred['yhat']
    
    print('========== RMSE ==========')
    print()
    #train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    #print(f'BATS Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'BATS Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()
    
    # Plot predictions compared vs y_test
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(df[[target]][:-period].index, df[[target]][:-period], color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='BATS predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(df.index[0::steps], fontsize=16);

    print('========== Predicted Value(s) ==========')
    print(y_pred)
    print('========== Test Value(s) ==========')
    print(y_test)
    
    return bats_pred

# TBATS Pipeline

def tbats(df, target, period, steps):

    # Instantiate
    estimator = TBATS()

    # Fit
    fitted_model = estimator.fit(df[target][:-period])

    # Forecast
    y_forecasted = fitted_model.forecast(steps=period)

    # Create Predicted DF
    tbats_pred = df[[target]][-period:]
    tbats_pred['yhat'] = y_forecasted

    # Calculating RMSE 
    y_train = df[target][:-period]
    #y_pred_train = forecast['yhat'][:-periods]

    y_test = tbats_pred[target]
    y_pred = tbats_pred['yhat']
    
    print('========== RMSE ==========')
    print()
    #train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    #print(f'BATS Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'TBATS Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()
    
    # Plot predictions compared vs y_test
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(df[[target]][:-period].index, df[[target]][:-period], color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='TBATS predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(df.index[0::steps], fontsize=16);

    print('========== Predicted Value(s) ==========')
    print(y_pred)
    print('========== Test Value(s) ==========')
    print(y_test)
    
    return tbats_pred

# Kalman Filter Pipeline

def kalman_filter(df, target, period, steps):
    
    # Extract target values
    observed_values = df[target][:-period].values

    # Instantiate a Kalman Filter model for time series prediction with dim_x = 2 (Position, Velocity)
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Define state transition matrix F and measurement matrix H for a simple linear model
    kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement matrix

    # Initialize state and covariance matrix
    kf.x = np.array([0, 0])  # Initial state [position, velocity]
    kf.P *= 1000  # Initial covariance matrix

    # Define process noise and measurement noise covariance matrices
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.01)
    kf.R = 0.1

    # Create arrays to store predictions and filtered state estimates
    predictions = []
    filtered_state_means = []

    # Number of future predictions to make
    num_predictions = period

    # Apply the Kalman Filter to observed data
    for z in observed_values:
        kf.predict()  # Predict the next state
        kf.update(z)  # Update the state estimate based on the measurement
        predictions.append(kf.x[0])  # Predicted position (state)
        filtered_state_means.append(kf.x[0])  # Filtered position (state)


    # Predict future values beyond the observed data
    for _ in range(num_predictions):
        kf.predict()  # Predict the next state without measurement
        predictions.append(kf.x[0])  # Predicted position (state)
        kf.update(z)
        filtered_state_means.append(kf.x[0])
        
    # Create DF with target, predictions & filteres state means
    kf_predictions = df[[target]]
    kf_predictions['predictions'] = predictions
    kf_predictions['filtered_state_means'] = filtered_state_means

    # Calculating RMSE 
    y_train = kf_predictions[target][:-period]
    y_pred_train = kf_predictions['predictions'][:-period]

    y_test = kf_predictions[target][-period:]
    y_pred = kf_predictions['predictions'][-period:]

    print('========== RMSE ==========')
    print()
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'Kalman Filter Train Root Mean Squared Error (RMSE): {train_rmse:.2f}')
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Kalman Filter Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print()
    
    # Plot predictions compared vs y_test
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(y_train.index, y_train, color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test.index, y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(y_test.index, y_pred, color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='Kalman Filter Predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(kf_predictions.index[0::steps], fontsize=16);

    print('========== Predicted Value(s) ==========')
    print(y_pred)
    print('========== Test Value(s) ==========')
    print(y_test)
    
    return kf_predictions

# LSTM Pipeline (For quarterly gdp data, for 1 year predictions)

def lstm(target_df, target, X_train, X_test, y_train, y_test, length=1, batch_size=1, features=1, epoch=1, periods=1, steps=1):

    # Setting random seeds for reproducibility
    tf.random.set_seed(123)

    # Create train sequence using Time Series Generator
    train_sequences = TimeseriesGenerator(X_train, y_train, length=length, batch_size=batch_size)

    # Create test sequence using Time Series Generator
    test_sequences = TimeseriesGenerator(X_test, y_test, length=length, batch_size=batch_size) 

    # Instantiate Model & Add layers
    model = Sequential()

    # Input Shape
    input_shape = (length,features)
    
    # Add 2 RNN (LSTM) layers
    model.add(LSTM(10, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(8, return_sequences=False))

    # Add Dense Layers with dropout of 0.5 at each layer
    model.add(Dense(8, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1)) 

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Instantiating EarlyStopping
    early_stop = EarlyStopping(monitor='val_root_mean_squared_error', 
                           min_delta=0, 
                           patience=3, 
                           mode='auto')

    # Fit Model
    lstm = model.fit(train_sequences, validation_data=test_sequences, epochs=epoch) 
    
    # Plotting training and test loss curves
    plt.figure(figsize=(30,15))
    plt.plot(lstm.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(lstm.history['val_root_mean_squared_error'], label='Test RMSE')
    plt.title(label='LSTM Loss Plot', fontsize=16)
    plt.legend();

    # Calculating RMSE 
    print('========== RMSE ==========')
    print()
    print(f"LSTM Train Root Mean Squared Error (RMSE): {lstm.history['root_mean_squared_error'][-1]:.2f}")
    print(f"LSTM Test Root Mean Squared Error (RMSE): {lstm.history['val_root_mean_squared_error'][-1]:.2f}")
    print()

    # Predictions & test values 
    y_pred = model.predict(test_sequences)
    y_test_calc = y_test[-periods:]
    
    # DF with results for reporting and plotting 
    pred_df = target_df[[target]][-periods:]
    pred_df['pred'] = y_pred
    
    # Plot predictions compared vs y_test
    plt.figure(figsize=(30,15))

    # Plot training data
    plt.plot(y_train, color='blue', label='y_train')

    # Plot testing data
    plt.plot(y_test, color='orange', label='y_test', alpha=0.7)

    # Plot predicted test values
    plt.plot(pred_df['pred'], color='green', label='y_pred', alpha=0.9, ls='--')

    plt.title(label='LSTM Predictions', fontsize=16)
    plt.legend(fontsize=20, loc='upper left')
    plt.xticks(target_df.index[0::steps], fontsize=16);

    print('========== Predicted Value(s) ==========')
    print(pred_df['pred'])
    print('========== Test Value(s) ==========')
    print(pred_df[target])

    return pred_df

















































