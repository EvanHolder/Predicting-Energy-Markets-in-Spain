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

def impute_mean_day(df, col):
    '''
    Find the days in the series that have Nans and impute the mean 
    (by hour) of the day before and after that are not Nan

    PARAMETERS
    ----------
    df: DataSeries,
        The dataframe that contains the series for which to impute
    col: str,
        The column name for which to impute the immediate mean
    '''
    # Get the dates of missing values
    dates = df.loc[df[col].isna()].index.normalize().unique()

    for date in dates:
        
        # Check if all hours in the day are empty, if not go to next date
        if df.loc[str(date.date()), col].isna().sum() < 24:
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