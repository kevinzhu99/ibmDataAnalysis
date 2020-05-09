import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.corr())

print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

#regress a fitted model: engine size and price
print(sns.regplot(x = 'engine-size', y = 'price', data = df))
print(plt.ylim(0,))

print(df[['engine-size', 'price']].corr())
print(sns.regplot(x = 'highway-mpg', y = 'price', data = df))
print(df[['highway-mpg', 'price']].corr())
print(sns.regplot(x="peak-rpm", y="price", data=df))
print(df[['peak-rpm','price']].corr())

print('\n\n\n\t\t\tnext\n\n\n')
#to display categorical variables, rather than quantitative variables, we use boxplots
#instead of regression plots
print(sns.boxplot(x = 'body-style', y = 'price', data= df))
print(sns.boxplot(x="engine-location", y="price", data=df))
print(sns.boxplot(x="drive-wheels", y="price", data=df))
print(df.describe(include=['object']))

#Don’t forget the method "value_counts" only works on Pandas series, not Pandas Dataframes.
# As a result, we only include one bracket "df['drive-wheels']" not two brackets
# "df[['drive-wheels']]".

print(df['drive-wheels'].value_counts())
#We can convert the series to a Dataframe as follows :
print(df['drive-wheels'].value_counts().to_frame())

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns = {'drive-wheels': "value_counts"}, inplace = True)
print(drive_wheels_counts)

#change the index to drive wheels: which is the row column
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

#repeat to the engine
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

#Grouping
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
#find the average price for each diff category of data
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

#create a pivot table for the grouped data to better represent the data
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
print(grouped_pivot)
#fill in all missing values with 0
grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot)

#Use the "groupby" function to find the average "price" of each car based on "body-style" ?
df_group2 = df[['body-style', 'price']]
group_test2 = df_group2.groupby(['body-style'], as_index = False).mean()
print(group_test2)

print('\n\n\n\n')
#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
#plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
#plt.show()

print(df.corr())

#lets calculate the pearson correlation coefficient and P-value
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)



