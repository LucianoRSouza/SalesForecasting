#!/usr/bin/env python
# coding: utf-8

# In[36]:


#!pip install xgboost


# In[72]:


import os
#!pip install tensorflow


# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# now let's have the csv file downloaded 
# In[80]:


store_sales = pd.read_csv('/Users/lucianosouza/Desktop/train.csv')
store_sales.head()


# In[81]:


#now we check for null values and the Dtype

store_sales.info()


# In[82]:


#now we drop the columns we won't be using

store_sales = store_sales.drop(['store', 'item'], axis = 1)


# In[83]:


store_sales.head()


# In[84]:


#now let's change the type of the date:

store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales.info()


# In[85]:


# Now we want to train the model to predict the sales for the next month, so we need to convert the date time to month and have the sales per month as well

store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()


# In[86]:


# now we can convert the resulting date column ro timestamp data type, as timestamp is one of the best types to work with date on a dataframe

monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales.head()


# In[87]:


# now let's visualize it:
plt.figure(figsize = (15, 5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title("Monthtly Customer Sales")
plt.show()


# In[88]:


#now let's create a column with the difference on the sales columns to make a sales stationary

monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
#let's also use the dropna to drop all the NaN values that can be returned form the diff function above.
monthly_sales = monthly_sales.dropna()


# In[89]:


#Now we need to train our model in order to predict the sales for the next 12 months so we need to prepare the supervised data set to fit into
#our ML model. So we'll use the previous 13 months to predict the next 12 months. For that we'll drop the columns date and sales,
#because we'll be only dealing with the stationary sales data and we'll use that to train our model as well as to reinforce the model later.

supervised_data = monthly_sales.drop(['date','sales'], axis=1)


# In[90]:


#Now we need to prepare it, where the previous 12 months sales will be our training data (input features) and the next 12 months
#sales will be the output for our supervised model

for i in range (1, 13):
    col_name = 'month_'+ str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
    
supervised_data = supervised_data.dropna().reset_index(drop=True)
supervised_data.head(10)


# In[91]:


#now let's split into train and test

train_data = supervised_data[:-12] #previous 12
test_data = supervised_data[-12:] #next 12

print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)


# In[92]:


#now we use the min and max to scale the feature values to -1 and 1
scaler = MinMaxScaler(feature_range=(-1,1))
#now we fit the train_data in the scaler
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


# In[93]:


#In supervised data frame the first column always correspond to the output and the remaining columns act as the input features

X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:0,:1]
y_train = y_train.ravel()
y_test = y_test.ravel()

#let's print the shape:
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[95]:


# Now let's mae a prediction df to merge the predicted sales prices of all trained algorithms. 
sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)


# In[96]:


predict_df.head()


# In[99]:


act_sales = monthly_sales['sales'][-13:].to_list()
#It'll store all the actual salesof the last 13 months


# In[101]:


# Ok, all above is just the preparation of the data, preprocessing of the data, nowe we'll make the forecasting using 
# Linear Regression, let's create a model and predict the output.
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)


# In[102]:


# As we have scaled the data before, we need to bring it back to the original format, so we simply reshape the predicted output
lr_pred = lr_pred.reshape(-1,1)
lr_pred_test_set = np.concatenate([lr_pred, X_test], axis=1)
lr_pred_test_set = scaler.inverse_transform(lr_pred_test_set)
#here we created a set matrix which contains the input features of the test data (X_test) and also the predicted output


# In[103]:


# now we need to calculate the predicted sales values

result_list = []
for index in range (0, len(lr_pred_test_set)):
    result_list.append(lr_pred_test_set [index][0] + act_sales[index])
    
lr_pred_series = pd.Series(result_list, name = "Linear Prediction")
predict_df = predict_df.merge(lr_pred_series, left_index=True, right_index=True)



# In[105]:


#since we have evaluated the predicted sales values of the test data, we can now evaluate various metrics for the linear regression
# model, we want to compare between the actual and predicted values, to see the deviations, differences and learn from that.
#We'll use the MSE, MAE and R2
lr_mse = np.sqrt(mean_squared_error(predict_df["Linear Prediction"], monthly_sales['sales'][-12:]))
lr_mae = mean_absolute_error(predict_df["Linear Prediction"], monthly_sales['sales'][-12:])
lr_r2 = r2_score(predict_df["Linear Prediction"], monthly_sales['sales'][-12:])
print('Linear Regression MSE: ', lr_mse)
print('Linear Regression MAE: ', lr_mae)
print('Linear Regression R2: ',lr_r2)


# In[107]:


# It looks pretty good, so now let's visualize the predicted agains the actual
plt.figure(figsize=(15,5))
#actual sales
plt.plot(monthly_sales['date'], monthly_sales['sales'])
#predicted sales
plt.plot(predict_df['date'], predict_df['Linear Prediction'])
plt.title('Customer sales Forecast using LR Model')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(['Actual sales', 'Predicted Sales'])
plt.show()


# In[114]:


# the result is this graph above with the actual sales in blue line from 2013 to 2018 and the predicted values in a yello 
#line from 2017 to 2018, which by the way is really close. 

# We can also visualize it in a bar chart style if you prefer. 

plt.figure(figsize=(15,5))
plt.bar(monthly_sales['date'], monthly_sales['sales'], width=10)
plt.plot(predict_df['date'], predict_df['Linear Prediction'], color = 'red')
plt.xlabel("Date")
plt.ylabel('Sales')
plt.title('Predictions vs Actual')
plt.show()


# In[ ]:




