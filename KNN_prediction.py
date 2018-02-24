# Kaggle Competition: Recruit Restaurant Visitor Forecasting
# You can find raw data in the url below
# https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting
# 2018.01.31

import pandas as pd
import numpy as np
from sklearn import *

# Read data from input folder
air_reserve = pd.read_csv('./input/air_reserve.csv')
air_store = pd.read_csv('./input/air_store_info.csv')
air_visit = pd.read_csv('./input/air_visit_data.csv')
date_info = pd.read_csv('./input/date_info.csv')
hpg_reserve = pd.read_csv('./input/hpg_reserve.csv')
hpg_store = pd.read_csv('./input/hpg_store_info.csv')
submission = pd.read_csv('./input/sample_submission.csv')
store_id = pd.read_csv('./input/store_id_relation.csv')

# Reservation data cleaning
# Seperate date from datetime and format in AirRegi reservation data
air_reserve['visit_datetime'] = pd.to_datetime(air_reserve.visit_datetime)
air_reserve['visit_date'] = air_reserve.visit_datetime.dt.date
air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve.reserve_datetime)
air_reserve['reserve_date'] = air_reserve.reserve_datetime.dt.date

# Generate new feature "duration"
air_reserve['duration'] = air_reserve.visit_datetime - air_reserve.reserve_datetime
air_reserve['duration_day'] = air_reserve.visit_date - air_reserve.reserve_date
air_reserve.duration = air_reserve.duration.astype('timedelta64[h]')
air_reserve.duration_day = air_reserve.duration_day.astype('timedelta64[D]')

# Calculate duration and visit by day
air_reserve_sum = air_reserve.groupby(['air_store_id','visit_date'], as_index=False)[['duration','duration_day','reserve_visitors']].sum().rename(columns={'duration': 'duration_sum', 'duration_day': 'duration_day_sum','reserve_visitors':'reserve_visitors_sum'})
air_reserve_mean = air_reserve.groupby(['air_store_id','visit_date'], as_index=False)[['duration','duration_day','reserve_visitors']].mean().rename(columns={'duration': 'duration_mean', 'duration_day': 'duration_day_mean','reserve_visitors':'reserve_visitors_mean'})
air_reserve = pd.merge(air_reserve_sum, air_reserve_mean, how='inner', on=['air_store_id','visit_date'])

# Seperate date from datetime and format in HotPepper reservation data
hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve.visit_datetime)
hpg_reserve['visit_date'] = hpg_reserve.visit_datetime.dt.date
hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve.reserve_datetime)
hpg_reserve['reserve_date'] = hpg_reserve.reserve_datetime.dt.date

# Calculate "duration" between reserve and visit
hpg_reserve['duration'] = hpg_reserve.visit_datetime - hpg_reserve.reserve_datetime
hpg_reserve['duration_day'] = hpg_reserve.visit_date - hpg_reserve.reserve_date
hpg_reserve.duration = hpg_reserve.duration.astype('timedelta64[h]')
hpg_reserve.duration_day = hpg_reserve.duration_day.astype('timedelta64[D]')

# Calculate duration and visit by day
hpg_reserve_sum = hpg_reserve.groupby(['hpg_store_id','visit_date'], as_index=False)[['duration','duration_day','reserve_visitors']].sum().rename(columns={'duration': 'duration_sum', 'duration_day': 'duration_day_sum','reserve_visitors':'reserve_visitors_sum'})
hpg_reserve_mean = hpg_reserve.groupby(['hpg_store_id','visit_date'], as_index=False)[['duration','duration_day','reserve_visitors']].mean().rename(columns={'duration': 'duration_mean', 'duration_day': 'duration_day_mean','reserve_visitors':'reserve_visitors_mean'})
hpg_reserve = pd.merge(hpg_reserve_sum, hpg_reserve_mean, how='inner', on=['hpg_store_id','visit_date'])

# Generate new feateure for stores based on duration
air_store_duration_sum = air_reserve.groupby(['air_store_id'])['duration_sum','duration_day_sum'].sum().reset_index()
air_store_duration_mean = air_reserve.groupby(['air_store_id'])['duration_mean','duration_day_mean'].mean().reset_index()
air_store_duration_sum.columns = ['air_store_id','duration_all_sum','duration_day_all_sum']
air_store_duration_mean.columns = ['air_store_id','duration_all_mean','duration_day_all_mean']
air_store_duration = pd.merge(air_store_duration_sum, air_store_duration_mean, how='inner', on='air_store_id')
hpg_store_duration_sum = hpg_reserve.groupby(['hpg_store_id'])['duration_sum','duration_day_sum'].sum().reset_index()
hpg_store_duration_mean = hpg_reserve.groupby(['hpg_store_id'])['duration_mean','duration_day_mean'].mean().reset_index()
hpg_store_duration_sum.columns = ['hpg_store_id','duration_all_sum','duration_day_all_sum']
hpg_store_duration_mean.columns = ['hpg_store_id','duration_all_mean','duration_day_all_mean']
hpg_store_duration = pd.merge(hpg_store_duration_sum, hpg_store_duration_mean, how='inner', on='hpg_store_id')


# Visit Data Cleaning
# Format the daytime feateure in visiting data
air_visit['visit_date'] = pd.to_datetime(air_visit.visit_date)
air_visit['visit_year'] = air_visit.visit_date.dt.year
air_visit['visit_month'] = air_visit.visit_date.dt.month
air_visit['visit_day'] = air_visit.visit_date.dt.day
air_visit['visit_dayOfWeek'] = air_visit.visit_date.dt.dayofweek

# Generate new feature based on visit
air_visit_visitors_month = air_visit.groupby(['air_store_id','visit_month']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()
air_visit_visitors_month.columns = ['air_store_id', 'visit_month', 'min_visitors_month', 'mean_visitors_month', 'median_visitors_month','max_visitors_month','count_visitors_month']
air_visit_visitors_dayOfWeek = air_visit.groupby(['air_store_id','visit_dayOfWeek']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()
air_visit_visitors_dayOfWeek.columns = ['air_store_id', 'visit_dayOfWeek', 'min_visitors_week', 'mean_visitors_week', 'median_visitors_week','max_visitors_week','count_visitors_week']
air_visit_visitors = pd.merge(air_visit, air_visit_visitors_month, how='inner', on=['air_store_id', 'visit_month'])
air_visit_visitors = pd.merge(air_visit_visitors, air_visit_visitors_dayOfWeek, how='inner', on=['air_store_id','visit_dayOfWeek'])
air_visit_visitors_sum = air_visit.groupby(['air_store_id'])['visitors'].sum().reset_index()
air_visit_visitors_sum.columns = ['air_store_id','visitors_sum']
air_visit_visitors_mean = air_visit.groupby(['air_store_id'])['visitors'].mean().reset_index()
air_visit_visitors_mean.columns = ['air_store_id','visitors_mean']
air_visit_visitors = pd.merge(air_visit_visitors, air_visit_visitors_sum, how='inner', on=['air_store_id'])
air_visit_visitors = pd.merge(air_visit_visitors, air_visit_visitors_mean, how='inner', on=['air_store_id'])

# Link with HotPepper store id to acquire HotPepper visiting data
hpg_visit_visitors = pd.merge(store_id, air_visit_visitors, how='inner', on=['air_store_id'])

# Store Info Cleaning
# Generate new feature to specify location of each store
air_store['LatAddLong'] = air_store.latitude + air_store.longitude
hpg_store['LatAddLong'] = hpg_store.latitude + hpg_store.longitude

# Categorize "area" and "genre" for stores
dummy_air = pd.get_dummies(air_store[['air_area_name', 'air_genre_name']], drop_first = True)
dummy_air_done = pd.merge(air_store, dummy_air , left_index=True, right_index=True)
air_store_info = pd.merge(air_visit_visitors, dummy_air_done, how='left', on='air_store_id')
dummy_hpg = pd.get_dummies(hpg_store[['hpg_area_name', 'hpg_genre_name']], drop_first = True)
dummy_hpg_done = pd.merge(hpg_store, dummy_hpg , left_index=True, right_index=True)
hpg_store_info = pd.merge(hpg_visit_visitors, dummy_hpg_done, how='inner', on='hpg_store_id')

# Date Data Cleaning
# Generate new flag as one day before holiday
date_info['holiday_shift'] = date_info['holiday_flg'].shift(-1)
date_info = date_info.fillna(0)
date_info.holiday_shift = date_info.holiday_shift.astype('int')
date_info['before_holiday'] = np.where((date_info['holiday_flg'] == 0) & (date_info['holiday_shift'] == 1) ,'1','0')
date_info = date_info.drop(['holiday_shift', 'day_of_week'], axis=1)
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])
date_info['calendar_date'] = date_info['calendar_date'].dt.date
date_info = date_info.rename(columns={'calendar_date':'visit_date'})

# Feature Combination
air_feature = pd.merge(air_store_info, air_reserve, how='left', on=['air_store_id', 'visit_date'])
air_feature['visit_date'] = pd.to_datetime(air_feature['visit_date'])
air_feature['visit_date'] = air_feature['visit_date'].dt.date
air_feature = pd.merge(air_feature, date_info, how='left', on='visit_date')
air_feature = pd.merge(air_feature, air_store_duration, how='left', on='air_store_id')
air_feature = air_feature.fillna(-1)
hpg_feature = pd.merge(hpg_store_info, hpg_reserve, how='left', on=['hpg_store_id', 'visit_date'])
hpg_feature['visit_date'] = pd.to_datetime(hpg_feature['visit_date'])
hpg_feature['visit_date'] = hpg_feature['visit_date'].dt.date
hpg_feature = pd.merge(hpg_feature, date_info, how='left', on='visit_date')
hpg_feature = pd.merge(hpg_feature, hpg_store_duration, how='left', on='hpg_store_id')
hpg_feature = hpg_feature.fillna(-1)

# Modeling
# Create two models due to the features are different in AirRegi and HotPepper
df_X_air = air_feature.drop(['air_store_id','air_area_name','air_genre_name','visit_date','visitors'], axis=1)
df_Y_air = air_feature.visitors
df_X_hpg = hpg_feature.drop(['air_store_id','hpg_store_id', 'hpg_area_name','hpg_genre_name','visit_date','visitors'], axis=1)
df_Y_hpg = hpg_feature.visitors

# Model object for AirRegi
X = df_X_air
y = np.log1p(df_Y_air.values)
mod_air = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)

#Seperate training set and testing set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 666)

#Modeling
mod_air.fit(X_train, y_train)

#Prediction
y_train_pred = mod_air.predict(X_train)
y_test_pred = mod_air.predict(X_test)

# Tuning parameter with trial and error
print('MSE Train : %.3f, Test : %.3f' % (metrics.mean_squared_error(y_train, y_train_pred), metrics.mean_squared_error(y_test, y_test_pred)))
print('RMSE Train : %.3f, Test : %.3f' % (metrics.mean_squared_error(y_train, y_train_pred)**0.5, metrics.mean_squared_error(y_test, y_test_pred)**0.5))

# Model object for HotPepper
X = df_X_hpg
y = np.log1p(df_Y_hpg.values)
mod_hpg = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)

# Seperate training set and testing set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 666)

# Modeling
mod_hpg.fit(X_train, y_train)

# Prediction
y_train_pred = mod_hpg.predict(X_train)
y_test_pred = mod_hpg.predict(X_test)
print('MSE Train : %.3f, Test : %.3f' % (metrics.mean_squared_error(y_train, y_train_pred), metrics.mean_squared_error(y_test, y_test_pred)))
print('RMSE Train : %.3f, Test : %.3f' % (metrics.mean_squared_error(y_train, y_train_pred)**0.5, metrics.mean_squared_error(y_test, y_test_pred)**0.5))

# Submission Data Cleaning
submission['air_store_id'] = submission['id'].map(lambda x: '_'.join(x.split('_')[:2]))
submission['visit_date'] = submission['id'].map(lambda x: str(x).split('_')[2])
submission['visit_date'] = pd.to_datetime(submission.visit_date)
submission['visit_year'] = submission.visit_date.dt.year
submission['visit_dayOfWeek'] = submission.visit_date.dt.dayofweek
submission['visit_month'] = submission.visit_date.dt.month
submission['visit_day'] = submission.visit_date.dt.day
submission['visit_date'] = submission['visit_date'].dt.date
submission = pd.merge(submission, store_id, how='left', on='air_store_id')

# AirRegi submission data cleaning
submission_air = pd.merge(submission, air_visit_visitors_month, how='left', on=['air_store_id', 'visit_month'])
submission_air = pd.merge(submission_air, air_visit_visitors_dayOfWeek, how='left', on=['air_store_id','visit_dayOfWeek'])
submission_air = pd.merge(submission_air, air_visit_visitors_sum, how='left', on=['air_store_id'])
submission_air = pd.merge(submission_air, air_visit_visitors_mean, how='left', on=['air_store_id'])
submission_air = pd.merge(submission_air, date_info, how='left', on=['visit_date'])
submission_air = pd.merge(submission_air, dummy_air_done, how='left', on=['air_store_id'])
submission_air = pd.merge(submission_air, air_reserve, how='left', on=['air_store_id', 'visit_date'])
submission_air = pd.merge(submission_air, air_store_duration, how='left', on=['air_store_id'])
submission_air = submission_air.fillna(-1)

# HotPepper submission data cleaning
submission_hpg = pd.merge(submission, air_visit_visitors_month, how='left', on=['air_store_id', 'visit_month'])
submission_hpg = pd.merge(submission_hpg, air_visit_visitors_dayOfWeek, how='left', on=['air_store_id','visit_dayOfWeek'])
submission_hpg = pd.merge(submission_hpg, air_visit_visitors_sum, how='left', on=['air_store_id'])
submission_hpg = pd.merge(submission_hpg, air_visit_visitors_mean, how='left', on=['air_store_id'])
submission_hpg = pd.merge(submission_hpg, date_info, how='left', on=['visit_date'])
submission_hpg = pd.merge(submission_hpg, dummy_hpg_done, how='left', on=['hpg_store_id'])
submission_hpg = pd.merge(submission_hpg, hpg_reserve, how='left', on=['hpg_store_id', 'visit_date'])
submission_hpg = pd.merge(submission_hpg, hpg_store_duration, how='left', on=['hpg_store_id'])
submission_hpg = submission_hpg.fillna(-1)

# Prediction feature cleaning
prediction_X_air = submission_air.drop(['id','air_store_id','hpg_store_id','air_area_name','air_genre_name','visit_date','visitors'], axis=1)
prediction_X_hpg = submission_hpg.drop(['id','air_store_id','hpg_store_id', 'hpg_area_name','hpg_genre_name','visit_date','visitors'], axis=1)

# AirRegi Prediction
submission_air.visitors = mod_air.predict(prediction_X_air)
submission_air['visitors'] = np.expm1(submission_air['visitors']).clip(lower=0.)
submission_air = submission_air.rename(columns={'visitors':'air_visitors'})

# HotPepper Prediction
submission_hpg.visitors = mod_hpg.predict(prediction_X_hpg)
submission_hpg['visitors'] = np.expm1(submission_hpg['visitors']).clip(lower=0.)
submission_hpg = submission_hpg.rename(columns={'visitors':'hpg_visitors'})

# Combine prediction data
submission_combine = pd.merge(submission, submission_air, how='left', on=['air_store_id', 'visit_date'])
submission_combine = pd.merge(submission_combine, submission_hpg, how='left', on=['air_store_id', 'visit_date'])
submission_result = submission_combine[['air_store_id','visit_date','air_visitors','hpg_visitors']]

# Format final submission data
submission_result = pd.merge(submission, submission_result, how='left', on=['air_store_id', 'visit_date'])
submission_result['visitors'] = submission_combine['air_visitors']
submission_result = submission_result[['id','visitors']]

# Export to csv
submission_result.to_csv('submission_result.csv', index=False)
