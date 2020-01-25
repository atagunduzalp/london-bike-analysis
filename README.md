# london-bike-analysis

In this project, I aim to predict bike rantal number according to wearther, season and days that given in dataset.
Python programming language used. 
https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset is used as a dataset. 


My dataset includes 17415 rows. I split it %20 for test and %80 for train. Next, you can find some sample charts to recognize and analyze data.

Here you can see distribution of the rental numbers with respect to hours

![hourly rental numbers](https://github.com/atagunduzalp/london-bike-analysis/blob/master/hourly-rental.png)

Here you can see distribution of the rental numbers with respect to months

![monthly rental numbers](https://github.com/atagunduzalp/london-bike-analysis/blob/master/monthly-rental.png)
 
These bar chats shows us the distribution of bike rentals with respect to temperature, wind speed,and humidity 

![bar_charts rental numbers](https://github.com/atagunduzalp/london-bike-analysis/blob/master/temp-hum-wind-distribution.png)

In this project, I tried several regression methods. These are AdaBoostRegressor, KNeighborsRegressor, LinearRegression, RandomForestRegressor and SVR. Results are different for each of these these models. Here, you can see R2 score and also RMSLE result of these models. 

R2 SCORE
![R2 SCORE](https://github.com/atagunduzalp/london-bike-analysis/blob/master/r2ScoresOfModels.png)


RMSLE
![RMSLE](https://github.com/atagunduzalp/london-bike-analysis/blob/master/RMSLE.png)

As you can see, RandomForestRegressor has the best r2 score and the least RMSLE score. In order to find out the fittest hyperparameters, RandomizedSearchCV applied with the selected values and parameters. 

Finally, as a result RandomizedSearchCV, best parameters selected with these parameters and values:


| Hyperparameter | n_estimators | max_features | max_depth | min_samples_leaf | min_samples_split | bootstrap |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| Value | 1800 | ‘auto’ | None | 2 |  2 |  True |  

