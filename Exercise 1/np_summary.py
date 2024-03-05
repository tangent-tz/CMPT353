import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']
minimum_row = np.argmin(np.sum(totals, axis=1))
print("Row with lowest total precipitation:")
print(minimum_row)
print("Average precipitation in each month:")
avg_precipitation_per_month = np.sum(totals, axis=0) / np.sum(counts, axis=0)
print(avg_precipitation_per_month)
print("Average precipitation in each city:")
avg_precipitation_per_city = np.sum(totals, axis=1) / np.sum(counts, axis=1)
print(avg_precipitation_per_city)
print("Quarterly precipitation totals:")
quarterly_precipitation_totals = np.sum(totals.reshape((totals.shape[0], 4, 3)), axis=2)
print(quarterly_precipitation_totals)

