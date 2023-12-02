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
file = os.path.join(cwd, "one_line_drawings.ndjson")
X = []
y = []

labels = {}
label_i = 0
count = 0

max_points = 918 # max number of points in a stroke

with open(file, "r") as f:
    for i,line in enumerate(f):
        line = json.loads(line)
        label = line["word"]
        if not len(labels):
            labels[label_i] = label
        elif labels[label_i] != label:
             label_i += 1
             labels[label_i] = label
             count = 0
        stroke = line["drawing"][0]
        X.append(stroke)
        y.append(label_i)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.00033, stratify=y) # random_state=42,

batch_size = 32
num_classes = label_i + 1
epochs = 50
nstrokes = 1
curr_slice = 0
slice_size = 1000
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv1D(32, kernel_size=4, activation='relu', input_shape=(max_points,max_points)))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print(X_test[0],X_train[0])

_X_test = []

for line in X_test:
    stroke = np.zeros((max_points,max_points), dtype=np.int8)
    x_stroke =line[0]
    y_stroke = line[1]
    stroke[0][:len(x_stroke)] = x_stroke
    stroke[1][:len(y_stroke)] = y_stroke
    _X_test.append(stroke)     

X_test = np.asarray(_X_test)
y_test = np.asarray(y_test)

total_loops = len(X_train) // slice_size

for i in range(total_loops): # max i = ~5580
    X_train_slice = np.zeros((slice_size,max_points,max_points))
    for j in range(slice_size): # X_train
        curr = i * slice_size + j
        stroke = np.zeros((max_points,max_points), dtype=np.int8)
        x_stroke = X_train[curr][0]
        y_stroke = X_train[curr][1]
        stroke[0][:len(x_stroke)] = x_stroke
        stroke[1][:len(y_stroke)] = y_stroke
        X_train_slice[j] = stroke      
    # X_test_slice = np.zeros(slice_size,max_points,max_points)
    # for j in range(len(slice_size)): # X_test
    #     curr = i * slice_size + j
    #     stroke = np.zeros(max_points,max_points)
    #     x_stroke = X_train[curr][0]
    #     y_stroke = X_train[curr][1]
    #     stroke[0][:len(x_stroke)] = x_stroke
    #     stroke[1][:len(y_stroke)] = y_stroke
    #     X_test_slice[curr] = stroke
    y_train_i = len(y_train) if len(y_train) < i+slice_size else i+slice_size
    model.fit(X_train_slice, y_train[i: y_train_i], callbacks=EarlyStopping(monitor='val_loss', patience=5), batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test[len(X_test)//2:], y_test[len(y_test)//2:]))
    print(str(i) + " out of " + str(total_loops))

score = model.evaluate(X_test[:len(X_test)//2], y_test[:len(y_test)//2])
print(score, num_classes, labels)



# model_file = open('model.pkl', 'wb') # open the file in write binary mode
# pickle.dump(model, model_file) # pickle the model object to the file
# model_file.close() # close the file

