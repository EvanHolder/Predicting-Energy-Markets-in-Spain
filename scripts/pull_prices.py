import requests
import datetime as dt
import pandas as pd

# Initialize dictionary to hold info
data = {'values':[], 'datetime':[]}

# Initialize request parameters
widget='precios-mercados-tiempo-real'
lang='en'
category='mercados'
time_trunc = 'hour'

# Request start and stop dates
start = dt.datetime(2019,1,1)
stop_pull = dt.datetime(2022,1,20)

while start < stop_pull:
    
    # calculate the stop time
    stop = start + dt.timedelta(hours=372)
    
    # if the stop time is at the end of the pull range, set it equal to last day
    if stop >= stop_pull:
        stop = stop_pull
    
    # Reformat date for api pull
    start_date = f'{start}'.replace(' ', 'T')[:-3]
    stop_date = f'{stop}'.replace(' ', 'T')[:-3]
    
    # Create get request
    url = f'https://apidatos.ree.es/{lang}/datos/{category}/{widget}?start_date={start_date}&end_date={stop_date}&time_trunc={time_trunc}'
    request = requests.get(url)
    json = request.json()
    
    # store request in data dictionary
    for val in json['included'][0]['attributes']['values']:
        data['values'].append(val['value'])
        data['datetime'].append(val['datetime'])
    
    # Go to the next hour
    start = stop + dt.timedelta(hours=1)
    
# Convert data to df
df = pd.DataFrame({'datetime':data['datetime'], 'price actual':data['values']})
df['datetime'] = pd.to_datetime(df['datetime'].apply(lambda x: x[:-16]))
df.set_index('datetime', inplace=True)
df.to_csv('../data/prices.csv')