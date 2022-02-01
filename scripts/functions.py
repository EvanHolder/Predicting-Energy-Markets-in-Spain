import numpy as np
import datetime as dt
import pandas as pd
from selenium import webdriver
import time

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