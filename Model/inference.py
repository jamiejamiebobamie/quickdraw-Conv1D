import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.callbacks import EarlyStopping
# from keras.layers import Conv1D, MaxPooling1D
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import pickle
import tensorflow
import numpy as np
import matplotlib.pyplot as plt

# labels = {'bat': 0, 'brain': 1, 'broom': 2, 'fire': 3, 'circle': 4, 'door': 5, 'feather': 6, 'frog': 7, 'hexagon': 8, 'hourglass': 9, 'hurricane': 10, 'key': 11, 'lightning': 12, 'moon': 13, 'mouth': 14, 'mushroom': 15, 'octagon': 16, 'octopus': 17, 'snake': 18, 'snowflake': 19, 'square': 20, 'star': 21, 'streetlight': 22, 'tornado': 23, 'triangle': 24}
# labels = {'bat': 0, 'brain': 1, 'broom': 2, 'fire': 3, 'circle': 4, 'door': 5, 'feather': 6, 'frog': 7, 'hexagon': 8, 'hourglass': 9, 'hurricane': 10, 'key': 11, 'lightning': 12, 'moon': 13, 'mouth': 14, 'mushroom': 15, 'octagon': 16, 'octopus': 17, 'snake': 18, 'snowflake': 19, 'square': 20, 'star': 21, 'streetlight': 22, 'tornado': 23, 'triangle': 24, 'axe': 25, 'castle': 26, 'cloud': 27, 'diamond': 28, 'dragon': 29, 'knee': 30, 'line': 31, 'skull': 32, 'squiggle': 33, 'stairs': 34, 'stitches': 35, 'zigzag': 36}
# labels = {'circle': 109333, 'bat': 10109, 'broom': 18180, 'door': 10784, 'feather': 9941, 'hexagon': 93543, 'hourglass': 41390, 'key': 27323, 'lightning': 84117, 'moon': 72575, 'mouth': 16476, 'mushroom': 30706, 'octagon': 89629, 'octopus': 8060, 'snake': 38486, 'square': 95601, 'star': 104394, 'streetlight': 27011, 'triangle': 90015, 'tornado': 6620, 'axe': 13309, 'castle': 20051, 'cloud': 40810, 'diamond': 35501, 'knee': 83030, 'line': 126574, 'squiggle': 74207, 'stairs': 102788, 'zigzag': 104508}
# labels = {'circle': 109333, 'bat': 10109, 'broom': 18180, 'door': 10784, 'feather': 9941, 'hexagon': 93543, 'hourglass': 41390, 'key': 27323, 'lightning': 84117, 'moon': 72575, 'mouth': 16476, 'mushroom': 30706, 'octagon': 89629, 'octopus': 8060, 'snake': 38486, 'square': 95601, 'star': 104394, 'streetlight': 27011, 'triangle': 90015, 'tornado': 6620, 'axe': 13309, 'castle': 20051, 'cloud': 40810, 'diamond': 35501, 'knee': 83030, 'line': 126574, 'squiggle': 74207, 'stairs': 102788, 'zigzag': 104508}
# labels = {'bat': 0, 'broom': 1, 'circle': 2, 'door': 3, 'hourglass': 4, 'lightning': 5, 'moon': 6, 'mushroom': 7, 'snake': 8, 'square': 9, 'star': 10, 'streetlight': 11, 'tornado': 12, 'triangle': 13, 'cloud': 14, 'diamond': 15, 'line': 16, 'zigzag': 17}
# labels = {0: 'circle', 1: 'door', 2: 'hourglass', 3: 'lightning', 4: 'moon', 5: 'mushroom', 6: 'square', 7: 'star', 8: 'tornado', 9: 'triangle', 10: 'diamond', 11: 'line'}

desired_labels = {
'circle': True,
'diamond': True,
'door': True,
'line': True,
'square': True,
'squiggle': True,
'star': True,
'triangle': True
}

cwd = os.getcwd()
file = os.path.join(cwd, "Dataset/one_line_drawings_pruned.ndjson")
X = []
y = []
label_to_i = {}
i_to_label = {}

label_i = 0
max_points = 40 # max number of points in a stroke

with open(file, "r") as f:
    for line in f:
        line = json.loads(line)
        label = line["word"]
        if label in desired_labels:
            stroke = np.zeros((max_points, 2)) # [[x,y], ...]
            x_stroke = np.asarray(line["drawing"][0][0])
            y_stroke = np.asarray(line["drawing"][0][1]) 
            f_data = [[x_stroke[i], y_stroke[i]] for i in range(len(x_stroke))]
            stroke[:len(f_data)] = f_data
            stroke /= 255
            X.append(stroke)
            if not len(label_to_i):         # first record
                label_to_i[label] = label_i
                i_to_label[label_i] = label
                j = label_i
            elif label not in label_to_i:   # new record
                label_i += 1
                label_to_i[label] = label_i
                i_to_label[label_i] = label
                j = label_i
            else:                           # old record
                j = label_to_i[label]
            y.append(j)

file = os.path.join(cwd, "Model\Pickled\model6.pkl")
with open(file, 'rb') as f:
    model = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, stratify=y)
predictions = []
y_pred = model.predict(np.asarray(X_test))
cm = confusion_matrix(np.asarray(y_test), np.argmax(y_pred, axis=1))

print(label_to_i)
print(i_to_label)
print("guess: " + i_to_label[np.argmax(y_pred, axis=1)[0]])
print("actual: " + i_to_label[y_test[0]])
print(X_test[0])

per_class_accuracies = {}

# Calculate the accuracy for each one of our classes
for idx, cls in enumerate(label_to_i):
    # True negatives are all the samples that are not our current GT class (not the current row) 
    # and were not predicted as the current class (not the current column)
    true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
    
    # True positives are all the samples of our current GT class that were predicted as such
    true_positives = cm[idx, idx]
    
    # The accuracy for the current class is the ratio between correct predictions to all predictions
    per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)

print(per_class_accuracies)
# {'bat': 0.9982832246063492, 'broom': 0.9963523302326441, 'circle': 0.981653954414324, 'door': 0.9969475300164858, 'feather': 0.9979155262388153, 'hexagon': 0.9668676368720531, 'hourglass': 0.9983405779045781, 'key': 0.9976580736556546, 'lightning': 0.9888543423775237, 'moon': 0.9897363086525097, 'mouth': 0.9989255815465126, 'mushroom': 0.9989096500747823, 'octagon': 0.9641784044439885, 'octopus': 0.9994404867128339, 'snake': 0.9913619560278635, 'square': 0.9944023180928626, 'star': 0.9985190103879569, 'streetlight': 0.9978040059367036, 'tornado': 0.9989077382981747, 'triangle': 0.998066556590818, 'axe': 0.9974273859450007, 'castle': 0.998923669769905, 'cloud': 0.9969965989494151, 'diamond': 0.9974694450303686, 'knee': 0.9889926275521421, 'line': 0.9921375000716917, 'squiggle': 0.9789825652345973, 'stairs': 0.9953868830457915, 'zigzag': 0.9840812734471435}

# disp is the ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[k for k in label_to_i])

# print(len(cm), len(labels))

# plot the confusion matrix
# disp.plot()

fig, ax = plt.subplots(figsize=(20,20))
disp.plot(ax=ax)

# show the plot
plt.show()




# # ---
# # see accuracy
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# predictions = []
# num_predict = 10000
# for i in range(num_predict):
#     predictions.append(model.predict(np.asarray([X_train[i]])))
# predictions = [np.argmax(p) for p in predictions]
# predictions = [True if predictions[i] == y_train[i] else False for i in range(len(predictions))]
# correct = filter(lambda x: x != False, predictions)
# print(len([c for c in correct]) / num_predict)
# # ---

# # ---
# # see incorrect guesses
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# predictions = []
# num_predict = 10000
# for i in range(num_predict):
#     predictions.append(model.predict(np.asarray([X_train[i]])))
# predictions = [np.argmax(p) for p in predictions]
# predictions = [True if predictions[i] == y_train[i] else { 'guessed': labels[predictions[i]], 'actual': labels[y_train[i]] } for i in range(len(predictions))]
# incorrect = filter(lambda x: x != True, predictions)
# print([inc for inc in incorrect])
# # ---

# ## ----
# file = os.path.join(cwd, "Model/model.pkl")
# with open(file, 'rb') as f:
#     model = pickle.load(f)
# # 3: fire
# test_sample = np.asarray([[[0.3254902,0.31764706],[0.30980392, 0.55294118],[0.30588235, 0.99607843],[0.25882353, 1.],[0.1372549,  0.96862745],[0.00392157, 0.96078431],[0.03921569, 0.88627451],[0.08235294, 0.68235294],[0.18039216, 0.4745098 ],[0.2, 0.36078431],[0.2, 0.2745098],[0.26666667, 0.2745098],[0.2745098, 0.25490196],[0.2745098,0.09803922],[0.25490196,0.01960784],[0.23529412, 0.0],[0.18431373, 0.14509804],[0.18039216, 0.21176471],[0.20392157, 0.23529412],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]])
# predictions = model.predict(test_sample)
# i = np.argmax(predictions)
# print(i)
# ## ----

# # print(model.summary())
