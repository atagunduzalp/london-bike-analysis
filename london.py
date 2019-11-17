from datetime import time

import pandas as pd
import numpy as np
import math, time, random, datetime
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing

import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn import model_selection, tree, preprocessing, metrics, linear_model

import seaborn as sns
import matplotlib.pyplot as plt
import missingno

data = pd.read_csv("london-bike.csv")
data = data.drop(columns=['t2'])
data.columns = [ 'date', 'count', 'temp', 'humadity', 'wind_speed', 'code', 'is_holiday', 'is_weekend', 'season']

data["hour"] = [t.hour for t in pd.DatetimeIndex(data.date)]
data["month"] = [t.month for t in pd.DatetimeIndex(data.date)]
print(data.head())

sns.factorplot(x="hour",y="count",data=data,kind='bar',size=5,aspect=1.5)
sns.factorplot(x="month",y="count",data=data,kind='bar',size=5,aspect=1.5)
sns.factorplot(x="is_weekend",y="count",data=data,kind='bar',size=5,aspect=1.5)
# plt.show(sns)
# extract and group hours

print(data.dtypes)
data['temp'] = pd.to_numeric(data['temp'],errors='coerce')
data['wind_speed'] = pd.to_numeric(data['wind_speed'],errors='coerce')
print(data.dtypes)
data = data.dropna()

"""
cor_mat= data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
plt.show(sns)
"""

# plt.show(missingno.matrix(data, figsize = (30,10)))
# print(data.max())

# Normalize features.
cols_to_norm = ['temp', 'humadity', 'wind_speed']
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# histogram
"""
fig,axes=plt.subplots(2,2)
axes[0,0].hist(x="temp",data=data,edgecolor="black",linewidth=2,color='red')
axes[0,0].set_title("Variation of temp")
axes[1,0].hist(x="wind_speed",data=data,edgecolor="black",linewidth=2,color='red')
axes[1,0].set_title("Variation of windspeed")
axes[1,1].hist(x="humadity",data=data,edgecolor="black",linewidth=2,color='red')
axes[1,1].set_title("Variation of humidity")
fig.set_size_inches(10,10)
plt.show()
"""
# plt.scatter(data['season'], data['count'])
# plt.show()

# create dummy variables

weekend_dummies = pd.get_dummies(data['is_weekend'],
                                     prefix='weekend')
holiday_dummies = pd.get_dummies(data['is_holiday'],
                                     prefix='holiday')
season_dummies = pd.get_dummies(data['season'],
                                     prefix='season')

weather_code_dummies = pd.get_dummies(data['code'],
                                     prefix='code')
data_dummy = pd.concat([data, holiday_dummies, weekend_dummies, season_dummies, weather_code_dummies], axis=1)
data_dummy = data_dummy.drop(columns=['is_weekend', 'is_holiday','season', 'code'])
print(data_dummy.head())

# seperate target column
X = data_dummy.drop(columns = ['count', 'date'])
y = data_dummy['count']
# train-test data splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# LinearRegressionModelCreation
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

# To retrieve the intercept:
print("intercept: " + str(regressor.intercept_))

#For retrieving the slope:
print("coef: " +str(regressor.coef_))
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_log_error
print("linear regression: " + str(r2_score(y_test, y_pred)))

#**********************************
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
models=[RandomForestRegressor(), AdaBoostRegressor(), BaggingRegressor(), SVR(), KNeighborsRegressor()]
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
rmsle=[]
rmsle_dict={}
r2 = []
r2_dict = {}

for model in range(len(models)):
    algo = models[model]
    algo.fit(X_train, y_train)
    test_pred = algo.predict(X_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred, y_test)))
    r2.append(r2_score(y_test, test_pred))

rmsle_dict = {'Modelling Algo': model_names, 'RMSLE': rmsle}
rmsle_frame = pd.DataFrame(rmsle_dict)
print(rmsle_frame)

r2_dict = {'Modelling Algo': model_names, 'R2Score': r2}
r2_frame=pd.DataFrame(r2_dict)
print(r2_frame)

sns.factorplot(y='Modelling Algo', x='R2Score', data=r2_frame, kind='bar', size=5, aspect=2)
sns.factorplot(y='Modelling Algo', x='RMSLE', data=rmsle_frame, kind='bar', size=5, aspect=2)
# plt.show(sns)
#**********************************

# RandomForest
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, y_train)
#
# print("best results: " + str(rf_random.best_params_))
# best results: {'max_features': 'auto', 'min_samples_split': 2, 'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 1800, 'bootstrap': True}

rmsle_rf=[]
rmsle_dict_rf={}
r2_rf= []
r2_dict_rf = {}

rf = RandomForestRegressor(n_estimators= 1000, max_features = 'auto', random_state=42,max_depth=None, min_samples_leaf=2, min_samples_split=2, bootstrap=True, )
rf.fit(X_train, y_train)
predicted_counts = rf.predict(X_test)
rmsle_rf.append(np.sqrt(mean_squared_log_error(predicted_counts, y_test)))
r2_rf.append(r2_score(y_test, predicted_counts))
print("r2 score: "+ str(r2_rf) + " rmsle: " + str(rmsle_rf))


# j = -1
# for i, v in y_test.iteritems():
#     j +=1
#     # print("this is i: " + str(i))
#     print("this is expected: " + str(v) + " this is predicted: " + str(y_pred[j]))
