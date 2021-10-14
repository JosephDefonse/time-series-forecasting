from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("PATH/train.csv") 
# print(df.head(10))

# Removes all the rows which have NaN - This drastically improves the model
df.dropna(inplace=True)

# Removing Outliers from the dataset
for i in range(0, len(df.columns)):
    if (df.columns[i] == 'Sales'):
        
        original = df.shape[0]
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        above = Q3[i-2] + 1.5 * IQR[i-2]
        below = Q1[i-2] - 1.5 * IQR[i-2]
        print(df.columns[i])
        print("High: "+str(above))
        print("Low: "+str(below))
        df.drop(df.loc[df[str(df.columns[i])] < below].index, inplace=True)
        df.drop(df.loc[df[str(df.columns[i])] > above].index, inplace=True)
        print("removed "+str(original-df.shape[0])+" rows")


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# Transforms the categorical variables to more numerical representation
# This transforms the Store categorical variable
df.Store = pd.Series(encoder.fit_transform(df.Store), index = df.index)

print(df.head(10))
print(df.dtypes)

df = df.reset_index()

df['Date'] = pd.to_datetime(df["Date"])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


from datetime import datetime

# Creating new column to distinguish weekend and weekdays as this variable would effectively predict item sales
daysval = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i in range(0, 7):
    arr = []
    for k in range(len(df)):
        year = df['Year'][k]
        month = df['Month'][k]
        day = df['Day'][k]
        if datetime(year, month, day).weekday() == i: # 5 is Satirday and 6 is Sunday 
            arr.append(1)
        else:
            arr.append(0)
    df[daysval[i]] = arr

# Creating new column to distinguish weekend and weekdays as this variable would effectively predict item sales
arr = []
for i in range(len(df)):
    year = df['Year'][i]
    month = df['Month'][i]
    day = df['Day'][i]
    if datetime(year, month, day).weekday() > 4: # 5 is Satirday and 6 is Sunday 
        arr.append(1)
    else:
        arr.append(0)
df['Weekend'] = arr

df.drop('Date', inplace=True, axis=1)

print(df.head(10))
# print(df.tail(10))

from sklearn.model_selection import train_test_split
# Including these variables as the x-axis variables in the regression
X = df.loc[:, ['Store', 'Item', 'Month', 'Day', 'Year', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Weekend']]
# Taking this variable as the y-axis variable in the regression
y = df.Sales
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=False)

from sklearn.ensemble import RandomForestRegressor
  
#create regressor object
model = RandomForestRegressor(n_estimators = 100, random_state = 0)
  
# fit the regressor with x and y data
print(model.fit(X_train, y_train))  

from sklearn.metrics import mean_squared_error

print(df.head(10))
print(df.dtypes)
print(mean_squared_error(pd.Series(model.predict(X_valid),index = X_valid.index),y_valid , squared=True))

# # from sklearn.linear_model import LinearRegression
# # model = LinearRegression()
# # print(model.fit(X, y)) # using the X_train and y_train variables to create/fit a regression model
# # print(pd.Series(model.predict(X), index = X.index))


# Inputting the testing data
X_valid=pd.read_csv("PATH/test.csv") 

X_valid.Store = pd.Series(encoder.fit_transform(X_valid.Store), index = X_valid.index)

X_valid['Date'] = pd.to_datetime(X_valid["Date"])
X_valid['Year'] = X_valid['Date'].dt.year
X_valid['Month'] = X_valid['Date'].dt.month
X_valid['Day'] = X_valid['Date'].dt.day

# Creating new column to distinguish weekend and weekdays as this variable would effectively predict item sales
daysval = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i in range(0, 7):
    arr = []
    for k in range(len(X_valid)):
        year = X_valid['Year'][k]
        month = X_valid['Month'][k]
        day = X_valid['Day'][k]
        if datetime(year, month, day).weekday() == i: # 5 is Satirday and 6 is Sunday 
            arr.append(1)
        else:
            arr.append(0)
    X_valid[daysval[i]] = arr

# Creating new column to distinguish weekend and weekdays as this variable would effectively predict item sales
arr = []
for i in range(len(X_valid)):
    year = X_valid['Year'][i]
    month = X_valid['Month'][i]
    day = X_valid['Day'][i]
    if datetime(year, month, day).weekday() > 4: # 5 is Saturday and 6 is Sunday 
        arr.append(1)
    else:
        arr.append(0)

X_valid['Weekend'] = arr

X_valid.drop('Date', inplace=True, axis=1)

# to get in the same order as the regression train data used before
X_valid = X_valid.loc[:, ['Store', 'Item', 'Month', 'Day', 'Year', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Weekend']]

print(X_valid.head(10))

output = pd.DataFrame({'id': X_valid.index+1, 'sales': model.predict(X_valid)})
output.to_csv('PATH', index=False)