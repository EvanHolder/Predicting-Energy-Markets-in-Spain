from bs4 import BeautifulSoup as BS
import pandas as pd
from functions import render_page, extract_table
import datetime as dt

# Create city code mappings
city_codes = {'LEZL':'seville',
              'LEBL':'barcelona',
              'LEBB':'bilboa',
              'LEVC':'valencia'}

# Create daterange for the pull
dates_stamp = pd.date_range(start=dt.datetime(2015,1,1), end=dt.datetime(2021,12,31),freq='D')

# Truncate dates for url
dates = [str(x)[:10] for x in dates_stamp]

# Create empty dictionary to store data for each city
city_data = {}

# Loop through each city code, collect data for each city
for code in city_codes.keys():
    
    # Create data dictionary to hold data for each data for this city
    data_1 = {}
    for date in dates:
        
        # Request page
        url = f'https://www.wunderground.com/history/daily/{code}/date/{date}'
        
        # Render page
        r = render_page(url)
        
        # Parse page and find table
        soup = BS(r, "html.parser")
        container = soup.find('lib-city-history-observation')
        table = container.find('tbody')

        # Extract this date's data from the table and add it to dictionary
        data_1[date] = extract_table(table)
    
    # Add all data to city_data dictionary
    city_data[code] = data_1

# Loop through each city in city_data
for code in city_data.keys():
    
    # Define empty lists for new dataframe cols
    dates_ =[]
    temps = []
    dew_points = []
    humidities = []
    winds = []
    wind_speeds = []
    wind_gusts = []
    pressures = []
    precips = []
    conditions = []
    
    # Loop through each day in the city
    for day_key in city_data[code].keys():
        
        # Loop through each row in the table and append info to col lists
        for ind, t in enumerate(city_data[code][day_key][0]):
            dates_.append(day_key+' '+city_data[code][day_key][0][ind])
            temps.append(city_data[code][day_key][1][ind])
            dew_points.append(city_data[code][day_key][2][ind])
            humidities.append(city_data[code][day_key][3][ind])
            winds.append(city_data[code][day_key][4][ind])
            wind_speeds.append(city_data[code][day_key][5][ind])
            wind_gusts.append(city_data[code][day_key][6][ind])
            pressures.append(city_data[code][day_key][7][ind])
            precips.append(city_data[code][day_key][8][ind])
            conditions.append(city_data[code][day_key][9][ind])
    
    # Create dataframe with all column information
    df = pd.DataFrame({'date':dates_,
                       'temp':temps,
                       'dew_point':dew_points,
                       'humidities': humidities,
                       'wind':winds,
                       'wind_speeds':wind_speeds,
                       'pressures':pressures,
                       'precips':precips,
                       'condition':conditions})
    df.to_csv(f'../data/{city_data[code]}.csv')