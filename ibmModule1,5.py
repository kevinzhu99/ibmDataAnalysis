import pandas as pd
import numpy as np
from IPython.display import display
from IPython.html import widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV



path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)
df.to_csv('module_5_auto.csv')

df = df._get_numeric_data()
print(df.head())

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

#Part 1 training and testing data
y_data = df['price']
#drop price data in x data
x_data=df.drop('price',axis=1)


#Now we randomly split our data into training and testing data  using the function
#the test_size parameter sets the proportion of data that is split into the testing set. In the above, the testing set is set to 10% of the total dataset.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

print("Question 1")
#Use the function "train_test_split" to split up the data set such that 40% of the data samples will be utilized for testing, set the parameter "random_state" equal to zero. The output of the function should be the following:  "x_train_1" , "x_test_1", "y_train_1" and  "y_test_1".
# Write your code below and press Shift+Enter to execute
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size = 0.4 , random_state = 0)

print("The number of test samples: " ,x_test1.shape[0])
print("The number of training samples: " , x_train1.shape[0])

#create a regression object
lre = LinearRegression()

print('THIS IS HOW YOU FIT THE MODEL, PLEASE LOOK AT (<CURRENT LINE> + 1)')
lre.fit(x_train[['horsepower']], y_train)
#calculate the coefficient of determination
lre.score(x_test[['horsepower']], y_test)
lre.score(x_train[['horsepower']], y_train)

#Find the R^2  on the test data using 90% of the data for training data
# Write your code below and press Shift+Enter to execute
x_train2,x_test2,y_train2,y_test2 = train_test_split(x_data, y_data, test_size = 0.1, random_state = 0)
lre.fit(x_train2[['horsepower']], y_train2)
lre.score(x_test2[['horsepower']], y_test2)

Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

print('We can use negative squared error as a score by setting the parameter (scoring) metric to (neg_mean_squared_error).')
print(-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error'))


print('Calculate the average R^2 using two folds, find the average R^2 for the second fold utilizing the horsepower as a feature :')
Rcross1 = cross_val_score(lre, x_data[['horsepower']], y_data, cv = 2)
print(Rcross1[1])
print('You can also use the function <cross_val_predict> to predict the output. The function splits up the data into the specified number of folds, using one fold to get a prediction while the rest of the folds are used as test data. First import the function:')
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
print(yhat[0:5])

#part 2: Over/Under fitting and model selection
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_train[0:5])

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test[0:5])

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
#print(DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title))

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
#DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
print(pr)

print("Now let's create a linear regression model <poly> and train it.")
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
print(yhat[0:5])

print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)

print("We will use the function <PollyPlot> that we defined at the beginning of the lab to display the training data, testing data, and the predicted function.")
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)

poly.score(x_train_pr, y_train)


#relook at the R^2 value
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

#part 3 Ridge Regression
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = 10 * np.array(range(0,1000))
for alfa in ALFA:
    RigeModel = Ridge(alpha=alfa)
    RigeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr, y_train))

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test, label='validation data  ')
plt.plot(ALFA,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

print("Perform Ridge regression and calculate the R^2 using the polynomial features, use the training data to train the model and test data to test the model. The parameter alpha should be set to 10")
# Write your code below and press Shift+Enter to execute
RigeModel = Ridge(alpha = 10)
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr,y_test)

print("GRID Search")

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
print(parameters1)

RR=Ridge()
print(RR)

#creating a grid object
Grid1 = GridSearchCV(RR, parameters1,cv=4)

Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
print("The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR as follows:")
BestRR=Grid1.best_estimator_
print(BestRR)
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)


print("Perform a grid search for the alpha parameter and the normalization parameter, then find the best values of the parameters")
# Write your code below and press Shift+Enter to execute
parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
Grid2 = GridSearchCV(Ridge(), parameters2,cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
Grid2.best_estimator_
