#import pandas library
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

#this file contains no headers
df = pd.read_csv(url, header = None)
print("The first 5 rows of the dataframe")
print(df.head(5))


#create headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

df.columns = headers
print(df.head(10))

#We can drop missing values along the column "price" as follows

df.dropna(subset = ['price'], axis = 0)
print(df)

#to print out the columns
print(df.columns)

#to save a dataset use: df.to_(<file extension>)(<file name>, <index = ?>)
df.to_csv("automobile.csv", index = False)

#view all the types for columns
print(df.dtypes)

#describe() - gives us the statisical summary of each column - df.describe() - only provides numeric
print(df.describe())

#describe(include = 'all') provides us with all information - even objects
print(df.describe(include = 'all'))

#you can select columns of a data frame by indicating the name of each column, e.g
print(df[['length', 'compression-ratio']].describe())

#info gives you the accurate information on each dataset
print(df.info)
