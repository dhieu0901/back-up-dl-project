import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data= pd.read_csv('./data/winedata.csv')
train_data, test_data=train_test_split(data, test_size=0.2, shuffle=True)
X_train=train_data.iloc[:, 0:13]
y_train=train_data.iloc[:, 13]
X_test=test_data.iloc[:, 0:13]
y_test=test_data.iloc[:, 13]
#print('y_train', np.unique(y_train, return_counts=True))
#print('y_test', np.unique(y_test, return_counts=True))
#print('y_test', y_test)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
def data_label(y):
    for i in range(len(y)):
        if (y[i]==1):
            y[i]=0
        elif (y[i]==2):
             y[i]=1
        else:
            y[i]=2
    return y

y_train=data_label(y_train)
y_test=data_label(y_test)
num_classes = 3
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

#model=MLPClassifier(hidden_layer_sizes=(300, 50), max_iter=1000, activation='logistic', random_state=1)

model = Sequential()
model.add(Dense(300, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(3, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=2, validation_data=(X_test, y_test))
# evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

def convert_label(y):
    for i in range(len(y)):
        if (y[i] == 0):
                y[i] = 1
        elif (y[i] == 1):
                y[i] = 2
        else:
                y[i] = 3
    return y

y_pred=convert_label(y_pred)
y_test=convert_label(y_test)
print('Accuracy:', accuracy_score(y_test, y_pred))