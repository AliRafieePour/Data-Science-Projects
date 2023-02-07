import pandas as pd
import numpy as np
import matplotlib
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler, Normalizer
import matplotlib.pyplot as plt
import keras


data = pd.read_csv('merged.csv')

data = data.fillna(0)
data = data.replace("\t", 0)
data = data.replace("\n", 0)
data = data.replace("", 0)
data = data.replace(" ", 0)
data = data.replace("  ", 0)
data = data.replace("   ", 0)
data = data.replace("    ", 0)


data['Date'] = pd.to_datetime(data['Date'])
data = data.groupby([data["Date"].dt.year, data["Date"].dt.month]).sum()

matplotlib.pyplot.plot(list(data['Market Demand'])[:-1])



Enrol_window = 48

sc = Normalizer( )
def load_data(datasetname, column, seq_len, normalise_window):
    # A support function to help prepare datasets for an RNN/LSTM/GRU
    data = datasetname.loc[:,column]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = sc.fit_transform(result)
        #result = normalise_windows(result)

    result = np.array(result)

    #Last 10% is used for validation test, first 90% for training
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    # A support function to normalize a dataset
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of <prediction_len> steps before shifting prediction run forward by <prediction_len> steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results(predicted_data, true_data): 
    fig = plt.figure(facecolor='white') 
    ax = fig.add_subplot(111) 
    ax.plot(true_data, label='True Data') 
    plt.plot(predicted_data, label='Prediction') 
    plt.legend() 
    plt.show() 

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show();


# feature_train, label_train, feature_test, label_test = load_data(data, 'Market Demand', Enrol_window, True)


# model = Sequential()
# model.add(LSTM(10, return_sequences=True, input_shape=(feature_train.shape[1],1)))
# model.add(Dropout(0.2))
# model.add(LSTM(10, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation = "linear"))


model = Sequential()
model.add(Dense(15, activation=('relu'), input_shape = (48,)))
model.add(Dropout(0.25))
model.add(Dense(15, activation=('relu')))
model.add(Dropout(0.25))
model.add(Dense(10, activation=('relu')))
model.add(Dropout(0.25))
model.add(Dense(10, activation=('relu')))
model.add(Dropout(0.25))
model.add(Dense(1))



opt = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='mse', optimizer=opt)


















data = list(data["Market Demand"])[:-1]



Years = []
for i in range(len(data)-48 + 1):
    Years.append(list(data[i:48+i]))

Years = np.array(Years)

# min_max_scaler = MinMaxScaler()
# twentyFour = min_max_scaler.fit_transform(twentyFour)
Years = sc.fit_transform(Years)
Y = []
for ind, ite in enumerate(Years):
    if (ind != 0):
        Y.append(ite[-1])

Y = np.array(Y).reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(Years[:-1], Y, test_size=0.2)

sc = Normalizer( )

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=36, epochs=5000)



start = Years[-1]
predictions = []
for i in range(20*12):
    start = start.reshape(1,-1)
    pr = model.predict(start)
    predictions.append(pr[0][0])
    start = start[:, 1:]
    start = np.append(start, pr)
    # start = sc.transform(start.reshape(1,-1))


























hist = model.fit(feature_train.reshape(198,12), label_train, batch_size=36, epochs=25, validation_data = (feature_test.reshape(22,12), label_test))


ac = list(data['Market Demand'])
tr = sc.transform(np.array(list(data['Market Demand'])).reshape(1, -1))

start = feature_test[-1]
predictions = []
for i in range(20*12):
    start = start.reshape(1,12,1)
    pr = model.predict(start)
    predictions.append(pr[0][0])
    start = start[:, 1:]
    start = np.append(start, pr)
    #start = sc.transform(start.reshape(1,-1))
