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

file = os.path.join(cwd, "Model\Pickled\model2.pkl")
with open(file, 'rb') as f:
    model = pickle.load(f)

spec = (tensorflow.TensorSpec((None, 40, 2), tensorflow.float32, name="input"),)
output_path = model.name + ".onnx"

# output_path = "sequential.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)

# 5 hexagon
test_sample = np.asarray(
[[[0.05490196, 0.0        ],
 [0.17254902, 0.02745098],
 [0.34901961, 0.04313725],
 [0.62352941, 0.04313725],
 [1.0,         0.64705882],
 [0.76862745, 0.77254902],
 [0.60784314, 0.91764706],
 [0.13333333, 0.90980392],
 [0.0,        0.54509804],
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

print(onnx_pred)
print('ONNX Predicted:', onnx_pred[0])
print(np.argmax(onnx_pred[0]))
