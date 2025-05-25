# Kaggle Predict Calorie Expenditure

## Problem Overview
The aim was to build a regression model capable of forecasting the number of calories burned following a workout, with scoring measured on root mean squared log error (RMSLE). <br>
Kaggle playground competition s5e5 (https://www.kaggle.com/competitions/playground-series-s5e5)

## Approach
1. **EDA**
- The dataset provided was large (75,000 entries) and clean - no missing values or erroneous data inputs - allowing me to focus on model development rather then cleaning and preprocessing data
- Features included numarical variables such as Height, Weight, Heart Rate, Duration, Age and one categorical feature, Gender.
- The target variable (Calories) was right skewed, with the majority of workouts resulting in a low-moderate number of calories burned (mean of 88, median of 77 and range of 1 - 314) as seen in the data.describe() and the histogram
- As expected, Calories was primarily correlated to Duration, Heart Rate and Body Temp, as seen in a correlation heatmap - this is where I would focus most of my feature engineering

2. **Feature Engineering**
- My primary focus was on model development rather than the data itself for this project, but I was able to try a range of concepts using real world concepts and feature correlations
- I created serveral new features using the highly correlated Duration, Heart Rate and Body Temp to try and gain some explanatory power with Weight, Height and Age and settled on the following due to improved correlations and explanatory power:
  - DHA (Duration * Heart Rate * Age) -> Younger people needed to workout longer and at higher intensity to burn more calories
  - HW_D ((Height + Weight) * Duration) -> People with a smaller mass needed to workout longer to burn calories
  - DHB (Duration * Heart Rate * Body Temp) -> This was a combination of the 3 highest explanatory variables and resulted in a 0.98 correlation
  - I added in an age category variable, and saw a slight difference in distributions for older age categories to younger, so used this feature to pick up some more subtle relationships
 
3. **Modelling**
- For the project I decided to use XGBRegressor due to its ability to handle non-linear relationships, a large dataset and a high number of hyperparameters I could tune in development
- I also converted the Calories target variable into the log1p - this is due to the fact that the Calorie distribution was skewed and also the scoring was based on RMSLE, so using log1p made sense
- For the model tuning I focused on the following hyperparameters and used GridSearchCV to find the optimal values:
  - Learning Rate
  - Lambda
  - Gamma
  - Max Depth
  - Min Child Weight
 - I also used Subsample and Tree Feature Sample of 0.8 to avoid overfitting to the training dataset
 - The optimal hyperparameter values I found for my data were: {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.5, 'max_depth': 6, 'min_child_weight': 0, 'reg_lambda': 6, 'subsample': 0.8}

4. **Evaluation** <br>

| Metric | RMSLE |
| --- | --- |
| Training RMSLE | 0.06130 |
| Testing RMSLE | 0.06500 |
| Public Leaderboard RMSLE | 0.06347|

The model showed strong generalisation to new data, validating my choice of hyperparameter values as well as the decision to log-transform the original training target variable.
