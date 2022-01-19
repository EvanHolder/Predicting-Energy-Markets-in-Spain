import numpy as np
import datetime as dt
import pandas as pd

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

# indices which have a value count of greater than one
def max_duplicated_indices(df, inplace=False):
    '''
    Given a dataframe which has duplicated indices, the function returns the max value
    for each column of the duplicated index.
    
    PARAMETERS
    ----------
    data: dataframe,
        The data which includes duplicated index
    inplace : bool, default False
        Whether to drop duplicates in place or to return a copy.
    
    RETURNS
    ----------
    data: dataframe with 
    '''
    if inplace:
        data = df
    else:
        data = df.copy()
    greater_than_one = data.loc[data.index.value_counts()>1].index.unique()
    for i in greater_than_one:
        cols = data.loc[i, data.loc[i].nunique()>1].columns
        for col in cols:
            val = data.loc[i,col].unique().max()
            data.loc[i,col] = val
    data.drop_duplicates(inplace=True)
    return data