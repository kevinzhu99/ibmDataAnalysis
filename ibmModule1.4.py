import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

file = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(file)
print(df.head())

#linear regression: creating fitted model
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat=lm.predict(X)
#outputs the value of the predicted values through slicing
print(Yhat[0:5])

#provides us the intercept of (a)
print(lm.intercept_)
#provides us the value of the slope(b)
print(lm.coef_)

#create a linear regression object
lm1 = LinearRegression()
print(lm1)

#train a model using 'engine size' and 'price'
lm1.fit(df[['engine-size']], df[['price']])
print(lm1)
#intercept and slope
print(lm1.intercept_)
print(lm1.coef_)

#multiple linear regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
print(lm.intercept_)
print(lm.coef_)

#model evaluation

#let's visualize Horsepower a potential predictor variable of price
width = 12
height = 10
print(plt.figure(figsize=(width, height)))
print(sns.regplot(x="highway-mpg", y="price", data=df))
print(plt.ylim(0,))

#inorder to get the plot to show, you must call the command
print(plt.show())

#compare this to the peak-rpm plot
print(plt.figure(figsize=(width, height)))
print(sns.regplot(x="peak-rpm", y="price", data=df))
print(plt.ylim(0,))
print(plt.show())


#residual plots
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()


#MLR
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


#polynomial regression and pipelines

#use the following function
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


x = df['highway-mpg']
y = df['price']


#Let's fit the polynomial using the function polyfit, then use the
# function poly1d to display the polynomial function

# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

#plot the graph
PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

#Pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
print(pipe)
print(pipe.fit(Z,y))
ypipe=pipe.predict(Z)
ypipe[0:4]

#Measures for In-Sample Evaluation

#calculate the R^2 value
lm.fit(X,Y)
print("The R squared value is: ", lm.score(X,Y))

Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# fit the model for Model 2: Linear Regression
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

#calculate the MSE
Y_predict_multifit = lm.predict(Z)

#compare predicted values with actual values
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#polynomial fitted model
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

#calculating mse
mean_squared_error(df['price'], p(x))

#prediction and decision making
new_input=np.arange(1, 100, 1).reshape(-1, 1)
#fitted model
lm.fit(X, Y)
#prediction
yhat=lm.predict(new_input)
yhat[0:5]
#plot the data
plt.plot(new_input, yhat)
plt.show()

