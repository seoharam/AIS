import pandas as pd
import numpy as np

path = './해미르호_항적'

df = pd.read_csv(path+'.csv', sep=',', skiprows=2, encoding='cp949')

df.columns = ['MMSI', 'date', 'longi', 'latit', 'SOG', 'COG', 'heading', 'voyage']

all_date = np.array(df['date'])

voyage_numbers = []
voyage_number = 1
stand_date = all_date[0][:10]
for date in all_date:
    date = date[:10]
    if stand_date == date:
        voyage_numbers.append(voyage_number)
    else:
        voyage_number += 1
        voyage_numbers.append(voyage_number)
        stand_date = date


df = pd.DataFrame(voyage_numbers)
df.to_csv(path+"_voyage_number.csv", index=False)