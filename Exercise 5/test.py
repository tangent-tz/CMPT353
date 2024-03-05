#%%
import pandas as pd
import glob
import os
#%%
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "data-*.csv"))
#%%
dataframes = {}
for csv_file in csv_files:
    key = os.path.basename(csv_file).split(".")[0]
    dataframes[key] = pd.read_csv(csv_file)
#%%
temp_df = pd.DataFrame(columns=['file', 'mean_x', 'mean_y', 'std_x', 'std_y', 'min_x', 'max_x', 'min_y', 'max_y', 'correration_coefficient', 'description'])
#%%
for key, df in dataframes.items():
    filename = key
    mean_x = df['x'].mean()
    mean_y = df['y'].mean()
    std_x = df['x'].std()
    std_y = df['y'].std()
    min_x = df['x'].min()
    max_x = df['x'].max()
    min_y = df['y'].min()
    max_y = df['y'].max()
    correration_coefficient = df['x'].corr(df['y'])
    description = ""
    dataframe = pd.DataFrame([[filename, mean_x, mean_y, std_x, std_y, min_x, max_x, min_y, max_y, correration_coefficient, description]], columns=['file', 'mean_x', 'mean_y', 'std_x', 'std_y', 'min_x', 'max_x', 'min_y', 'max_y', 'correration_coefficient', 'description'])
    temp_df = temp_df._append(dataframe)

temp_df.reset_index(drop=True, inplace=True)

temp_df.at[0, 'description'] = "Strong positive correlation between X and Y"
temp_df.at[1, 'description'] = "Strong positive correlation between X and Y"
temp_df.at[2, 'description'] = "Very strong positive correlation between X and Y"
temp_df.at[3, 'description'] = "Very weak negative correlation between X and Y"
temp_df.at[4, 'description'] = "Very weak negative correlation between X and Y"
temp_df.at[5, 'description'] = "Strong positive correlation between X and Y"

#save print to summart.txt
with open('summary.txt', 'w') as f:
    for index, row in temp_df.iterrows():
        f.write("File: " + row['file'] + "\n")
        f.write("\tMean X: " + str(row['mean_x']) + "\n")
        f.write("\tMean Y: " + str(row['mean_y']) + "\n")
        f.write("\tStandard deviation X: " + str(row['std_x']) + "\n")
        f.write("\tStandard deviation Y: " + str(row['std_y']) + "\n")
        f.write("\tRange X: " + str(row['min_x']) + " - " + str(row['max_x']) + "\n")
        f.write("\tRange Y: " + str(row['min_y']) + " - " + str(row['max_y']) + "\n")
        f.write("\tCorrelation coefficient: " + str(row['correration_coefficient']) + "\n")
        f.write("\tDescription: " + row['description'] + "\n\n")

