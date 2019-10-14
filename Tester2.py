#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
#%matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')

#print the head
print(df.head())

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')


#plt.show()


#Use averages of consecutive days to predict what's gonna happen
#Does this to compute the average of the next day,
#Following that it includes that computed average to add on
#To all the days and compute the average of the next day, etc


#create dataframe with date and the target variable (Closing price)

# data = df.sort_index(ascending=true, axis=0)
# new_data = pd.dataframe(index=range(0, len(df)), columns=['date', 'close'])
#
# for i in range(0, len(data)):
#     new_data['date'][i] = data['date'][i]
#     new_data['close'][i] = data['close'][i]
#
# #set last year's data in validation to last 4 years data before that in train
#
# train = new_data[:987]
# valid = new_data[987:]
#
# new_data.shape, train.shape, valid.shape
# ((1235, 2), (987, 2), (248, 2))
#
# train['date'].min(), train['date'].max(), valid['date'].min(), valid['date'].max()
#
#
#
#
# #make predictions
# preds = []
# for i in range(0,248):
#     a = train['close'][len(train)-248+i:].sum() + sum(preds)
#     b = a/248
#     preds.append(b)
#
#
#
# #calculate rmse
# rms=np.sqrt(np.mean(np.power((np.array(valid['close'])-preds),2)))
# rms
# print(rms)
# #plot
# valid['predictions'] = 0
# valid['predictions'] = preds
# plt.plot(train['close'])
# plt.plot(valid[['close', 'predictions']])
# plt.show()

#So far what we have may not be the best model,
#The most effective method is LSTM long short term memory.

 #LSTM is able to store past information that is important, and forget the information that is not. LSTM has three gates:

#The input gate: The input gate adds information to the cell state
#The forget gate: It removes the information that is no longer required by the model
#The output gate: Output Gate at LSTM selects the information to be shown as output

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms
print(rms)

#for plotting
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

plt.show()