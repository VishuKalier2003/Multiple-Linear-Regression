import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame()
data['Depend'] = (1, 3, 4, 6, 7, 8, 9, 10, 10, 11, 12, 14, 15, 19, 17, 18, 19, 18, 19, 20)
data['v1'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
data['v2'] = (3, 4, 6, 7, 8, 9, 10, 9, 11, 12, 13, 10, 11, 15, 16, 18, 19, 20, 21, 22)

d1 = data[['v1', 'v2']]
d2 = data[['Depend']]
x = np.array(d1)
y = np.array(d2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)
regression = LinearRegression()
regression.fit(x, y)
ypred = regression.predict(xtest)
ypred1 = regression.predict(xtrain)
r = regression.score(xtrain, ytrain) * 100
print("The Multiple Regression Accuracy on the Training set is : ",regression.score(xtrain, ytrain)*100,'%')
print("The Multiple Regression Accuracy on the Testing set is : ",np.sqrt(regression.score(xtest, ytest))*100,"%")
print("The Multiple Regression Accuracy on the Predicted set is :",np.mean(regression.score(xtest, ypred))*100,"%")
print("The Mean Squared Error for Training Set is : ",mean_squared_error(ytrain, ypred1),"%")
print("The Mean Squared Error for Testing Set is : ",mean_squared_error(ytest, ypred),"%")