import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import time

# api call: api.openweathermap.org/data/2.5/forecast?lat=1&lon=1&appid=9ad764c407e86718dc8b7430e3a11a16&units=metric


city_list = ['Bucharest', 'Copenhagen', 'Reykjavik', 'Doha', 'Lima', 'Brussels',
             'Athens', 'Washington D.C.', 'Ankara', 'Luxembourg', 'Gibraltar',
             'Kathmandu', 'Ottawa', 'Kyiv', 'London', 'Tokyo', 'Seoul',
             'Amsterdam']
temp_table = []
for city in city_list: # making a small table for each city in particular
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat=1&lon=1&appid=9ad764c407e86718dc8b7430e3a11a16&units=metric&q={city}"
    response = requests.get(url)
    response_json = response.json()
    df = pd.json_normalize(response_json['list'])
    df['weather'] = df['weather'].str[0].str['main']
    relevant_columns = ['dt_txt', 'main.temp', 'weather', 'wind.speed', 'main.humidity', 'main.feels_like', 'clouds.all']
    df = df[relevant_columns]
    df['City'] = response_json['city']['name']
    temp_table.append(df)
    time.sleep(1)
master_table = pd.concat(temp_table)
master_table = master_table.rename(columns={
    'dt_txt': 'Date & Time',
    'main.temp': 'Temperature',
    'main.feels_like': 'Feels Like',
    'clouds.all': 'Cloud Coverage %',
    'main.humidity': 'Humidity',
    'wind.speed': 'Wind Speed',
    'weather': 'Weather'
})
train_set, test_set = train_test_split(master_table, train_size=510, random_state=42)
train_set.to_csv("train_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)
print(master_table)