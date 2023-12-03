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

file = os.path.join(cwd, "Model\Pickled\model4.pkl")
with open(file, 'rb') as f:
    model = pickle.load(f)

spec = (tensorflow.TensorSpec((None, 40, 2), tensorflow.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)

# 7 star
test_sample = np.asarray([
[[0.27058824, 0.4       ],
 [0.33333333, 0.29019608],
 [0.42745098, 0.04705882],
 [0.45098039, 0.01176471],
 [0.47843137, 0.0        ],
 [0.50196078, 0.02352941],
 [0.58823529, 0.25490196],
 [0.66666667, 0.38431373],
 [0.9372549,  0.37254902],
 [1.0,         0.38431373],
 [0.90196078, 0.46666667],
 [0.72941176, 0.65490196],
 [0.67058824, 0.61960784],
 [0.74117647, 0.83921569],
 [0.56470588, 0.77254902],
 [0.4745098,  0.69019608],
 [0.42352941, 0.60784314],
 [0.23137255, 0.80784314],
 [0.17647059, 0.84313725],
 [0.18039216, 0.78431373],
 [0.21176471, 0.68627451],
 [0.31372549, 0.47058824],
 [0.0,         0.46666667],
 [0.31764706, 0.29411765],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ],
 [0.0,         0.0        ]]], dtype=np.float32)

onnx_pred = m.run(output_names, {"input": test_sample})
labels = {0: 'circle', 1: 'door', 2: 'hourglass', 3: 'lightning', 4: 'moon', 5: 'mushroom', 6: 'square', 7: 'star', 8: 'tornado', 9: 'triangle', 10: 'diamond', 11: 'line'}
print('ONNX Predicted:', onnx_pred[0])
print(labels[np.argmax(onnx_pred[0])])
