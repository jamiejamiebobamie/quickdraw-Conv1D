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
file = os.path.join(cwd, "Dataset/one_line_drawings.ndjson")
# files = glob.glob(relative_path)
data = []

with open(file, "r") as f:
    for line in f:
        data.append(line)


labels = {'circle': 0, 'bat': 0, 'broom': 0, 'door': 0, 'feather': 0, 'hexagon': 0, 'hourglass': 0, 'key': 0, 'lightning': 0, 'moon': 0, 'mouth': 0, 'mushroom': 0, 'octagon': 0, 'octopus': 0, 'snake': 0, 'square': 0, 'star': 0, 'streetlight': 0, 'triangle': 0, 'tornado': 0, 'axe': 0, 'castle': 0, 'cloud': 0, 'diamond': 0, 'knee': 0, 'line': 0, 'squiggle': 0, 'stairs': 0, 'zigzag': 0}
file = os.path.join(cwd, "Dataset\one_line_drawings_pruned.ndjson")
# add to  dataset of one line drawings
with open(file, "w") as f:
    for i,d in enumerate(data):
        d = json.loads(d)
        strokes = d["drawing"]
        label = d["word"]
        points = strokes[0][0]
        if len(strokes) == 1 and d["recognized"] and len(points) < 41 and label in labels:
            json.dump(d, f)
            f.write("\n")
            labels[label] += 1

print(labels)

#     for d in data:
#         d = json.loads(d)
#         drawing = d["drawing"]
#         category = d["word"]
#         isR = d["recognized"]
#         if len(drawing) == 1 and isR:
#             json.dump(d, f)
#             f.write("\n")
#             print(category)


# [ # drawing
#     [ # stroke1
#         [96, 82, 57, 33, 6, 10, 2, 0, 13, 49, 70, 116, 116, 109, 104, 97, 111, 132, 143, 148, 151, 149, 143, 147, 135, 126, 125, 136, 136, 134, 124],
#         [9, 22, 32, 34, 30, 64, 100, 129, 126, 106, 98, 95, 50, 15, 4, 6, 6, 0, 2, 17, 40, 111, 182, 249, 254, 255, 250, 186, 152, 125, 100]
#     ]
# ]   

# print(categories, count)

# max_points = float("-inf")

# n_points = []

# with open(file, "r") as f:
#     for i,line in enumerate(f):
#         line = json.loads(line)
#         label = line["word"]
#         n = len(line["drawing"][0][0])
#         n_points.append(n)
#         max_points = max(max_points, n)

# median = sorted(n_points)[len(n_points) // 2]
# acc = 0
# for n in n_points:
#     acc += n
# mean = acc // len(n_points)

# print(max_points, median, mean) # 40 20 20, total: 999150