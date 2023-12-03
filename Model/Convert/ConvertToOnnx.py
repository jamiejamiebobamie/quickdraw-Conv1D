import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.callbacks import EarlyStopping
# from keras.layers import Conv1D, MaxPooling1D
import json
from sklearn.model_selection import train_test_split
import os
import pickle
import tensorflow
# import tensorflow.compat.v1 as tf
import numpy as np
import tf2onnx
import onnxruntime as rt

cwd = os.getcwd()

file = os.path.join(cwd, "Model\Pickled\model3.pkl")
with open(file, 'rb') as f:
    model = pickle.load(f)

spec = (tensorflow.TensorSpec((None, 40, 2), tensorflow.float32, name="input"),)
output_path = model.name + ".onnx"

# output_path = "sequential.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)

# 4 hourglass
test_sample = np.asarray([[
 [0.05098039, 1.0],
 [0.2,        0.95686275],
 [0.39215686, 0.93333333],
 [0.52941176, 0.92941176],
 [0.82745098, 0.95686275],
 [0.83137255, 0.94509804],
 [0.70588235, 0.77647059],
 [0.50588235, 0.63137255],
 [0.42352941, 0.5372549 ],
 [0.2745098,  0.41176471],
 [0.19215686, 0.29803922],
 [0.06666667, 0.0745098 ],
 [0.44705882, 0.0        ],
 [0.83137255, 0.01176471],
 [0.97254902, 0.00392157],
 [0.96862745, 0.03529412],
 [0.93333333, 0.10980392],
 [0.82352941, 0.25098039],
 [0.4745098,  0.55294118],
 [0.14509804, 0.80392157],
 [0.05490196, 0.88627451],
 [0.0,         0.97647059],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0],
 [0.0,0.0]]], dtype=np.float32)

onnx_pred = m.run(output_names, {"input": test_sample})
labels = {0: 'bat', 1: 'broom', 2: 'circle', 3: 'door', 4: 'hourglass', 5: 'lightning', 6: 'moon', 7: 'mushroom', 8: 'snake', 9: 'square', 10: 'star', 11: 'streetlight', 12: 'tornado', 13: 'triangle', 14: 'cloud', 15: 'diamond', 16: 'line', 17: 'zigzag'}
print('ONNX Predicted:', onnx_pred[0])
print(labels[np.argmax(onnx_pred[0])])
