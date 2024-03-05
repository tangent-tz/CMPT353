import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

print("City with lowest total precipitation:")
minimum_precipitation_city = totals.sum(axis=1)
minimum_precipitation_city_min = minimum_precipitation_city.idxmin()
print(minimum_precipitation_city_min)
print("Average precipitation in each month:")
avg_precipitation_per_month = totals.sum(axis=0) / counts.sum(axis=0)
print(avg_precipitation_per_month)
print("Average precipitation in each city:")
avg_precipitation_per_city = totals.sum(axis=1) / counts.sum(axis=1)
print(avg_precipitation_per_city)

