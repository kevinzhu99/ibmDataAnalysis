import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
import numpy as np


filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)
print(df.head(5))

#how to replace missing values
df.replace("?", np.nan, inplace = True)
print(df.head(5))

#evaluating missing data - two methods (extension): 1. .isnull() 2. .notnull()
missing_data = df.isnull()
print(missing_data.head(5))

#count the missing values in each column - value_counts() counts the number
#of true values which are missing values
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')

#calculate missing values
avg_norm_loss = df['normalized-losses'].astype("float").mean(axis = 0)
print("Average of normalized-losses:", avg_norm_loss)

#replace Nan by the value of normalized losses
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace=True)


#caluclate the mean value for 'bore' column
avg_bore = df['bore'].astype('float').mean(axis =0)
print("Average of bore", avg_bore)
df['bore'].replace(np.nan, avg_bore, inplace = True)

#quiz question
avg_stroke = df['stroke'].astype('float').mean(axis = 0)
df['stroke'].replace(np.nan, avg_stroke, inplace= True)
print("The average of stroke is:" , avg_stroke)

print("\n\tFind the number of missing values for num-of-doors\n")
print(df['num-of-doors'].value_counts())
#we can use .idmax() to caluclate the most common type automatically
print(df['num-of-doors'].value_counts().idxmax())

#replacement procedure
print(df['num-of-doors'].replace(np.nan, "four", inplace = True))

#drop the whole row with NaN in "price" column
print(df.dropna(subset=["price"], axis=0, inplace=True))

# reset index, because we droped two rows
print(df.reset_index(drop=True, inplace=True))
print(df.head())
print(df.dtypes)

#convert data types properly
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print(df.dtypes)


print('\n\n\ntesttest')
#data transformation
print(df.head())
df['city-L/100km'] = 235/df['city-mpg']
#rename something
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
print(df.head())


# replace (original value) by (original value)/(maximum value) to normalize
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
print(df[['length', 'width', 'height']].head())

print(df.columns)
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

#treat this as a dictionary
dummy_variable_1.rename(columns={'fuel-type-gas':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
print(dummy_variable_1.head())

print('\n\n\ntesttesttest\n\n\n')
# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
print(df.head())

dummy_variable_2 = pd.get_dummies(df["aspiration"])
print(dummy_variable_2)
dummy_variable_2.rename(columns={'std':"aspiration-std", "turbo": 'aspiration-turbo'},inplace = True )
print(dummy_variable_2.head())

# Write your code below and press Shift+Enter to execute

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "fuel-type" from "df"
df.drop("aspiration", axis = 1, inplace=True)
df.to_csv('clean_df.csv')