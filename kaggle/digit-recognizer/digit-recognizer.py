import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dense

train = pd.read_csv('./train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('./test.csv').values).astype('float32')

y_train = to_categorical(labels) 

scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

kaynet = Sequential()
kaynet.add(Dense(512, activation='relu', input_shape=(28 * 28, )))
kaynet.add(Dense(10, activation='softmax'))

kaynet.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
               
               
kaynet.fit(X_train, y_train, epochs=10, batch_size=256)

print('Generating test predictions')
preds = kaynet.predict_classes(X_test)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "kay-digit-recognizer.csv")