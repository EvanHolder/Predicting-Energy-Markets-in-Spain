import numpy as np
import datetime as dt
import pandas as pd
from selenium import webdriver
import time
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.layers import TimeDistributed
from tensorflow import keras

def equal(v1,v2):
    '''
    Check if one value is equal to another
    
    PARAMETERS
    ----------
    v1: int, float, nan, None str
        First value to compare
    v2: int, float, nan, None str
        Second value to compare
    
    RETURNS
    ----------
    bool:
        True where v1 and v2 are the same number, float, string,
        or are both None or nan
    '''
    if v1 == None or v2 == None:
        return v1 == None and v2 == None
    elif type(v1) == str or type(v2) == str:
        return str(v1) == str(v2)
    elif np.isnan(v1) and np.isnan(v2):
        return True
    elif type(v1) == type(v2):
        return v1 == v2
    else:
        return False

# Define function returns closest values in the series that are not Nan
def find_nearest(series, time):
    '''
    Find the nearest element in the series that is not the same as the
    element at the specified time in the series
    
    PARAMETERS
    ----------
    series: DataSeries,
        The series which to search
    time: datetime, str,
        The index for which to find the nearest values to
    
    RETURNS
    ----------
    val_minus : object
        Element in the series which comes before the specified time index,
        not equal to the time index
    val_plus : object
        Element in the series which comes after the specified time index,
        not equal to the time index
        
    '''
    val_minus, val_plus = series[time], series[time]
    delta = 1
    while equal(val_minus, series[time]) or equal(val_plus, series[time]):
        if (equal(val_minus, series[time]) and not 
            equal(series[time - dt.timedelta(hours=delta)], series[time])):
            val_minus = series[time - dt.timedelta(hours=delta)]
        if (equal(val_plus, series[time]) and not 
            equal(series[time + dt.timedelta(hours=delta)], series[time])):
            val_plus = series[time + dt.timedelta(hours=delta)]
        delta += 1
    return val_minus, val_plus

# Given series and specific time, impute the mean time for most immediately known adjacent data
def impute_immediate_mean(series, time):
    '''
    Impute the mean of nearest elements before and after a specified index
    in the timeseries
    
    PARAMETERS
    ----------
    series: DataSeries,
        The series which to impute
    time: datetime, str,
        The index for which to impute the immediate mean
    
    RETURNS
    ----------
    mean: float
        The mean value of the element before and after series[time]
    '''
    val_minus, val_plus = find_nearest(series, time)
    return round(np.mean([val_minus, val_plus]),1)

def impute_mean_day(df, col, threshold):
    '''
    Find the days in the series that have Nans and impute the mean 
    (by hour) of the day before and after that are not Nan

    PARAMETERS
    ----------
    df: DataSeries,
        The dataframe that contains the series for which to impute
    col: str,
        The column name for which to impute the immediate mean
    threshold: int
        The number of values present in a single day.
        Anything day with less than threshold Nan values not be imputed
    '''
    # Get the dates of missing values
    dates = df.loc[df[col].isna()].index.normalize().unique()

    for date in dates:
        
        # Check if threshold hours in the day are empty, if not go to next date
        if df.loc[str(date.date()), col].isna().sum() < threshold:
            continue
        
        # Get the data point 24 hours before and after and average them
        delta_minus = dt.timedelta(hours=24)
        delta_plus  = dt.timedelta(hours=24)
        cur_date = date
        
        # Check if the day before is nan, adjust if not
        while np.isnan(df.loc[cur_date-delta_minus, col]):
            delta_minus += dt.timedelta(hours=24)
        while np.isnan(df.loc[cur_date+delta_plus, col]):
            delta_plus  += dt.timedelta(hours=24)
        
        # Get the data points for the first full day before and after and average them
        for i in range(1,25):
            df.loc[cur_date,col] = np.mean([df.loc[cur_date-delta_minus, col],
                                            df.loc[cur_date+delta_plus, col]])
            cur_date += dt.timedelta(hours=1)
            
def daylight_savings_shift(data, date_col):
    '''
    Shift dates columns to line up with Spain daylight savings period.
    
    PARAMETERS
    ----------
    df: dataframe,
        The data for which to shift the date column
    date_col: string,
        The name of the date column
    
    RETURNS
    ----------
    dt_: DataSeries,
        The realigned dataseries shifted according to CET daylight savings time.
    '''
    df = data.copy()
    # Define daylight savings each year
    ds = {2015:[dt.datetime(2015,3,29), dt.datetime(2015,10,25)],
          2016:[dt.datetime(2016,3,27), dt.datetime(2016,10,30)],
          2017:[dt.datetime(2017,3,26), dt.datetime(2017,10,29)],
          2018:[dt.datetime(2018,3,25), dt.datetime(2018,10,28)],
          2019:[dt.datetime(2019,3,31), dt.datetime(2019,10,27)],
          2020:[dt.datetime(2020,3,29), dt.datetime(2020,10,25)],
          2021:[dt.datetime(2021,3,28), dt.datetime(2021,10,31)]}

    # Reset the index
    df.reset_index(inplace=True)

    # Create datetime col based on the date col
    df['dt_'] = pd.to_datetime(df[date_col])
    
    for i, date in enumerate(df.dt_):
        if (date >= ds[date.year][0]) and (date <= ds[date.year][1]):
            std_hour = df.loc[i, 'date'][11:]
            if std_hour == '11:00 PM':
                df.loc[i, 'dt_'] = df.loc[i, 'dt_']-dt.timedelta(hours=23)
            else:
                df.loc[i, 'dt_'] = df.loc[i, 'dt_']+dt.timedelta(hours=1)
    return df.dt_

def clean_weather(data):
    '''
    Given scraped weather data, clean up the indices and columns.
    
    PARAMETERS
    ----------
    data: dataframe,
        The data for which to clean up, scraped
    RETURNS
    ----------
    data: DataFrame,
        The cleaned dataframe
    
    '''
    df = data.copy()
    # Drop any row that is not on the top of the hour
    df['top_of_hour']= df['date'].apply(lambda x: x[-5:-3]=='00')
    df = df.loc[df.top_of_hour].copy()
    df.drop(columns='top_of_hour', inplace=True)
    
    # Reset index
    df.reset_index(inplace=True)

    # Shift date column to appropriate daylight savings time, and make datetime
    df['date'] = daylight_savings_shift(df, 'date')
    
        # Reset index
    df.set_index('date',inplace=True)

    # Truncate temperature, dew_point, humidities, wind_speeds, pressures, precips data
    df.temp = df.temp.apply(lambda x: int(x[:-3]))
    df.dew_point = df.dew_point.apply(lambda x: int(x[:-3]))
    df.humidities = df.humidities.apply(lambda x: int(x[:-3]))
    df.wind_speeds = df.wind_speeds.apply(lambda x: int(x[:-5]))
    df.pressures = df.pressures.apply(lambda x: float(x[:-4]))
    df.precips = df.precips.apply(lambda x: float(x[:-4]))
    return df


def render_page(url):
    '''
    Render page source with a three second delay given a url.
    Libraries: webdriver from selenium, time
    
    PARAMETERS
    ----------
    url: str,
        The url from which to render the page source
    
    RETURNS
    ----------
    r: html-page source
    '''
    driver = webdriver.Chrome('chromedriver')
    driver.get(url)
    time.sleep(3)
    r = driver.page_source
    driver.quit()
    return r

def extract_table(table):
    '''
    Extracts the summary weather table for a specific day from wundergroun.com
    Libraries: webdriver from selenium, time, BeautifulSoup4, pandas
    
    PARAMETERS
    ----------
    table: BeautifulSoup4 container with header tbody
        
    RETURNS
    ----------
    day: list of nested lists,
        Table data for each column in the wunderground webpage
    '''
    day = [[] for i in range(10)]
    for row in table.find_all('tr', class_='ng-star-inserted'):
        for i, col in enumerate(row.find_all('td', class_='ng-star-inserted')):
            day[i].append(col.text)
    return day

def split_data(data, test_year, target):
    '''
    Splits the input dataframe into train and rest years. 
    Libraries: pandas
    
    PARAMETERS
    ----------
    data: dataframe,
        The data which to split into training and testing sets
    test_year: int,
        The year to use for testing set
    target: str, list,
        The target variable or list of target variables
        
    RETURNS
    ----------
    X_train: dataframe,
        X training data
    y_train: dataseries
        y training data
    X_test: dataframe,
        X testing data
    y_test: dataseries,
        y testing data
    '''
    # Get list of columns and name train_cols
    train_cols = data.columns.to_list()
    
    # Remove target cols from train_cols list
    if type(target) is list:
        for col in target:
            train_cols.remove(col)
    else:
        train_cols.remove(target)
        
    # Divide dataset into training and validation
    X_train = data.loc[:str(test_year-1), train_cols]
    y_train = data.loc[:str(test_year-1), target]
    X_test = data.loc[str(test_year), train_cols]
    y_test = data.loc[str(test_year), target]
    return X_train, y_train, X_test, y_test


def SMAPE(y_true, y_pred):
    '''
    Calculates the symmetric absolute mean.
    
    Libraries: keras.backend
    
    PARAMETERS
    ----------
    y_true: list-like,
        The true values in a series, numpy array, or list-like
    y_pred: list-like,
        The predictions in a series, numpy array, or listi-like
        
    RETURNS
    ----------
    SMAPE: float,
        The symmetric mean absolute percentage error
    '''
    return 100 * K.mean(abs(y_pred - y_true)/((abs(y_true)+abs(y_pred))/2))

def sMAPE(y_true, y_pred, d_type=None):
    if d_type == 'tensor':
        return 100 * K.mean(abs(y_pred - y_true)/((abs(y_true)+abs(y_pred))/2))
    else:
        return 100/(len(y_true)) * (abs(y_pred - y_true)/((abs(y_true)+abs(y_pred))/2)).sum()
    
def r2(y_true, y_pred):
    '''
    Calculates r-squared (coefficient of determination squared).
    
    Libraries: numpy
    
    PARAMETERS
    ----------
    y_true: list-like,
        The true values in a series, numpy array, or list-like
    y_pred: list-like,
        The predictions in a series, numpy array, or listi-like
        
    RETURNS
    ----------
    SMAPE: float,
        The r-squared value
    '''
    return np.corrcoef(y_true, y_pred)[0][1]**2

def compute_metrics(model, param_dict, train, test):
    '''
    Compute SMAPE and r-squared for a given training and/or testing dataset.
    
    Libraries: numpy
    
    PARAMETERS
    ----------
    model: estimator,
        The estimator used to predict y_pred
    param_dict: dictionary or string,
        The specific parameters used in fitting the estimator
    train: tuple,
        The dataset used to train the estimator in (predictors, targets) format
    test: tuple,
        The dataset used to test the estimator in (predictors, targets) format
        
    RETURNS
    ----------
    SMAPE: list,
        List containing the parameters used to fit the estimator, and estimators computed metrics, format: [params, SMAPE_train, SMAPE_test, r2_train, r2_test]
    '''
    # Convert data into arrays if not already
    X_train, y_train = np.array(train[0]), np.array(train[1])
    X_test, y_test = np.array(test[0]), np.array(test[1])
    
    # Predict 
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    # Compute sMAPE
    sMAPE_train = sMAPE(y_train.flatten(), preds_train.flatten())
    sMAPE_val = sMAPE(y_test.flatten(), preds_test.flatten())

    # Compute r2
    r2_train = r2(y_train.flatten(), preds_train.flatten())
    r2_val = r2(y_test.flatten(), preds_test.flatten())
    
    return [param_dict, 
            round(sMAPE_train,3), 
            round(sMAPE_val,3), 
            round(r2_train,3), 
            round(r2_val,3)]


def to_supervised(train, n_input, n_out=7, stride=1):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :-1])
            y.append(data[in_end:out_end, -1])
        # move along one time step
        in_start += stride
    return np.array(X), np.array(y)

def window_gen(data, input_window, output_window, stride):
    '''
    Yields train and test samples of the given provided datasets, at specified input and out lengths, and specified strides
    
    Libraries: pandas
    
    PARAMETERS
    ----------
    train: tuple,
        Tuple of length 2, which provides the training features and training targets respectively
    test: tuple,
        Tuple of length 2, which provides the testing features and testing targets respectively
    input_window: int,
        Length of the sequence of input data
    output_window: int,
        Length of the sequence of output data
    stride, int
        Number of steps to move between first sample and second sample
    YIELDS
    ----------
    windows: array or dataframe,
        Samples of specified input_window, output wind
    '''
    # Define X_train, y_train, X_test, y_test
    X, y = data[0], data[1]
    
    # Compute number of samples 
    n = len(X)/stride
    
    # If the input_window is greater than a day, X_train
    if input_window > 24:
        n_add = input_window - 24
        X = X.iloc[:n_add].append(X)
    else:
        n_add = 0
    
    i=0
    for i in range(0, len(X)-n_add, stride):
        yield X.iloc[i:i+input_window].to_numpy(), y.iloc[i:i+output_window].to_numpy()

def resample(data, input_window, output_window, stride):
    '''
    Generates a new dataset in batches of samples of given input_window, output_window and stride
    
    Libraries: numpy, pandas
    
    PARAMETERS
    ----------
    data: tuple,
        Tuple of length 2 (predictors, targets), which provides the input data to be batched into specified input / output windows
    input_window: int,
        The length of the input_sequence for each batch of predictors
    output_window: int,
        Length of the out_put sequence for each batch of targets
    stride, int
        Number of steps to move between first sample and second sample
    RETURNS
    ----------
    x_data, y_data: array, array,
        Batched predictors and associated targets of specified input/output windows 
    '''
    win = window_gen((data[0], data[1]), input_window=input_window, output_window=output_window, stride=stride)
    
    n = int(len(data[0])/stride)
    X_data = np.array([])
    y_data = np.array([])
    
    for i in range(n):
        X_sample, y_sample = next(win)
        X_data = np.append(X_data, X_sample)
        y_data = np.append(y_data, y_sample)
        
    # Reshape
    X_data = X_data.reshape(n,input_window,len(data[0].columns))
    y_data = y_data.reshape(n, output_window)
    
    return X_data, y_data

def compile_fit(nn,
                train,
                validation,
                patience=10,
                metric=SMAPE,
                loss = tf.keras.metrics.mean_absolute_error,
                batch_size = None, 
                verbose='auto', 
                learning_rate=.001):
    '''
    Compiles and fits a neural network.
    
    Libraries: keras, tensorflow
    
    PARAMETERS
    ----------
    nn: estimator,
        Keras model
    train: tuple, (predictors, targets),
        The predictors and targets the estimator will be fit to
    validation: tuple, (predictors, targets),
        The predictors and targets the keras will use to evaluate the EarlyStopping method
    metric, keras metric class
        Metric for estimator to be compiled with
    loss: keras loss,
        Loss function to be used to fit the estimator
    batch_size: int,
        Size of the input samples
    verbose: boolean,
        Output progress, yes or no
    learning_rate, float
        The learning rate specified in Adam optimizer
    RETURNS
    ----------
    nn: keras model,
        Compiled and fit neural network
    '''
    # Create early stopping point
    metric.name='SMAPE'
    callback = keras.callbacks.EarlyStopping(
        patience=patience,
        monitor='val_'+metric.name,
        mode='min',
        restore_best_weights=True
    )
    # Compile the model
    nn.compile(
        loss=loss, 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metric
    )
    if type(train[0]) is list:
        train_x = [X_data for X_data in train[0]]
        val_x = [X_data for X_data in validation[0]]
    else:
        train_x = [train[0]]
        val_x = [validation[0]]
    # Fit the model
    history = nn.fit(
        x = train_x,
        y = train[1],
        batch_size=batch_size,
        epochs = 200,
        callbacks=[callback],
        validation_data=(val_x, validation[1]),
        verbose=verbose
    )
    return nn

def set_param(model, param, value):
    '''
    Sets the parameters for a specified model
    
    Libraries: keras
    
    PARAMETERS
    ----------
    model: estimator,
        Keras model
    param: str,
        The name of the parameter to set
    value: float, int, str,
        The value of the specified parameter to be set
    RETURNS
    ----------
    nn: keras model,
        keras estimator with the specified parameter set
    '''
    
    if param == 'max_depth':
        model.set_params(max_depth=value)
    if param == 'gamma':
        model.set_params(gamma=value)
    if param == 'min_child_weight':
        model.set_params(min_child_weight=value)
    if param == 'subsample':
        model.set_params(subsample=value)
    if param == 'colsample_bytree':
        model.set_params(colsample_bytree=value)
    if param == 'reg_alpha':
        model.set_params(reg_alpha=value)
    if param == 'reg_lambda':
        model.set_params(reg_lambda=value)
    model.set_params(random_state=17)
    return model


def plot_metric_range(model, train, test, param, range_):
    '''
    Fits an estimator, computes metrics, and plots the results over a metric range.
    
    Libraries: sklearn, XGBoost, numpy, pandas
    
    PARAMETERS
    ----------
    model: estimator,
        Estimator with fit and predict methods
    train: tuple (predictors, targets),
        Tuple containing the predictors and targets used to be fit the estimator and compute metrics
    test: tuple (predictors, targets),
        Tuple containing the predictors and targets of the test set used to compute metrics
    param: str,
        The name of the parameter to be set
    range_: list,
        The range of values that the specified parameter will be set to before training each estimator
    '''
    
    #Create lists to hold metrics for plotting
    sMAPE_train, sMAPE_val, r2_train, r2_val = [],[],[],[]
    
    # For each element in metric range, fit model, compute metrics, and add to respective lists
    for i in range_:
        x = model
        x = set_param(x, param, i)
        x.fit(train[0], train[1])
        metrics = compute_metrics(x, 'None',(train[0], train[1]), (test[0], test[1]))
        sMAPE_train.append(metrics[1])
        sMAPE_val.append(metrics[2])
        r2_train.append(metrics[3])
        r2_val.append(metrics[4])
    
    # Plot the metrics, one plot for train, one for test
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].plot(range_, sMAPE_train, label='sMAPE_train')
    ax[0].plot(range_, sMAPE_val, label='sMAPE_val');
    ax[0].set(xlabel=f'{param}', ylabel='sMAPE score', title=f'{param} versus sMAPE');
    ax[0].legend();
    ax[1].plot(range_, r2_train, label='r2_train');
    ax[1].plot(range_, r2_val, label='r2_val');
    ax[1].legend();
    ax[1].set(xlabel=f'{param}', ylabel='r2 score', title=f'{param} versus r2');
    
def ensemble_nn(models):
    '''
    PARAMETERS
    ----------
    models: list,
        List containing trained models to use in ensemble
    RETURNS
    ----------
    ensemble: keras model,
        Trained model combining all input models into a single output model.
    '''
    # Get models in list
    models = [model for model in models]

    # Rename layers 
    for i, model in enumerate(models):
        for i2, layer in enumerate(model.layers):
            layer.trainable = False
            layer._name = f'ensemble_{i}_{i2}_{layer.name}'
    
    # Define multi-headed input
    ensemble_visible = [model.input for model in models]
    
    # Concatenate merge output from each model
    ensemble_outputs = [model.output for model in models]
    merge = layers.merge.concatenate(ensemble_outputs)
    hidden = layers.Dense(24, activation='relu')(merge)
    output = TimeDistributed(layers.Dense(1))(hidden)
    ensemble = keras.Model(inputs=ensemble_visible, outputs=output)   
    return ensemble
    