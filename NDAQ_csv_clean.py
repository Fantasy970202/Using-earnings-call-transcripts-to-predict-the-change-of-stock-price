import pandas as pd
import numpy as np
data = pd.read_csv('NASDAQ_HistoricalData.csv')
for i in range(len(data['Date'])):
    date = data.iloc[i]['Date'].split('-')
    date[0], date[1] = date[1], date[0]
    y = '-'.join(date)
    data.loc[i, 'Date'] = y
new = data.drop(["Unnamed: 6","Unnamed: 7", "Unnamed: 8"], axis=1)
print(new)
new.to_csv('NASDAQ_17_22.csv')
