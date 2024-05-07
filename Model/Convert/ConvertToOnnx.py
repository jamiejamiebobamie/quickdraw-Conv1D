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

file = os.path.join(cwd, "Model\Pickled\model7.pkl")
with open(file, 'rb') as f:
    model = pickle.load(f)

spec = (tensorflow.TensorSpec((None, 40, 2), tensorflow.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)

# 2 diamond
test_sample = np.asarray([
[[0.3254902,  0.01568627],
 [0.30588235, 0.08235294],
 [0.10196078, 0.36078431],
 [0.0,         0.54509804],
 [0.23921569, 0.80784314],
 [0.44313725, 1.0        ],
 [0.44705882, 0.98039216],
 [0.89411765, 0.56470588],
 [0.87843137, 0.57254902],
 [0.80784314, 0.5254902 ],
 [0.58431373, 0.34901961],
 [0.23137255, 0.0        ],
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
labels = {0: 'circle', 1: 'star', 2: 'diamond', 3: 'line', 4: 'squiggle'}

print('ONNX Predicted:', onnx_pred[0])
print(labels[np.argmax(onnx_pred[0])])
