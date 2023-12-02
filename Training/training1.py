import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
import numpy as np
import glob
import os
import json
from sklearn.model_selection import train_test_split
import pickle

cwd = os.getcwd()
file = os.path.join(cwd, "Dataset/one_line_drawings_pruned.ndjson")
X = []
y = []
label_to_i = {}
label_i = 0
max_points = 40 # max number of points in a stroke

with open(file, "r") as f:
    for line in f:
        line = json.loads(line)
        label = line["word"]
        stroke = np.zeros((max_points, 2)) # [[x,y], ...]
        x_stroke = np.asarray(line["drawing"][0][0])
        y_stroke = np.asarray(line["drawing"][0][1]) 
        f_data = [[x_stroke[i], y_stroke[i]] for i in range(len(x_stroke))]
        stroke[:len(f_data)] = f_data
        stroke /= 255
        X.append(stroke)
        if not len(label_to_i):         # first record
            label_to_i[label] = label_i
            j = label_i
        elif label not in label_to_i:   # new record
            label_i += 1
            label_to_i[label] = label_i
            j = label_i
        else:                           # old record
            j = label_to_i[label]
        y.append(j)


batch_size = 64
num_classes = len(label_to_i)
epochs = 30
nstrokes = 1

print(num_classes, label_to_i)
# 29 {'bat': 0, 'broom': 1, 'circle': 2, 'door': 3, 'feather': 4, 'hexagon': 5, 'hourglass': 6, 'key': 7, 'lightning': 8, 'moon': 9, 'mouth': 10, 'mushroom': 11, 'octagon': 12, 'octopus': 13, 'snake': 14, 'square': 15, 'star': 16, 'streetlight': 17, 'tornado': 18, 'triangle': 19, 'axe': 20, 'castle': 21, 'cloud': 22, 'diamond': 23, 'knee': 24, 'line': 25, 'squiggle': 26, 'stairs': 27, 'zigzag': 28}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

half = len(X_test) // 2

model = Sequential()
model.add(Conv1D(40, kernel_size=3, activation='relu', input_shape=(max_points,2)))
model.add(MaxPooling1D(pool_size=3, strides=1))
model.add(Dropout(0.25))
model.add(Conv1D(80, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Dropout(0.25))
model.add(Conv1D(160, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=1, strides=1))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(640, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(X_train, y_train, callbacks=EarlyStopping(monitor='val_loss', patience=3), batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test[half:], y_test[half:]))
score = model.evaluate(X_test[:half], y_test[:half])
print(score)

model_file = open('model.pkl', 'wb')
pickle.dump(model, model_file)
model_file.close()

model.save(os.path.join(cwd, model.name))
