import os
import json
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dropout, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

split_test_path=r"path/to/test/sequences"
# Load model
model_path=r"path/to/model"
model=load_model(model_path)

# Load data
def load_data(split_path):
  X = []
  Y = []
  seqs_list=os.listdir(split_path)
  random.shuffle(seqs_list)
  for seq in seqs_list:
    seq_file_path=os.path.join(split_path,seq)
    with open(seq_file_path, "r") as f:
      data = json.load(f)
      X.append(data['features'])
      Y.append(data['class'])
  X = np.array(X)
  Y = np.array(Y)
  return X,Y


X_test, y_test = load_data(split_test_path)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 12, 2))

y_test_predicted=model.predict(X_test)
y_test_predicted= np.argmax(y_test_predicted,-1)


confusion_matrix = metrics.confusion_matrix(y_test, y_test_predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Neutral", "Agressive","Victim"])

cm_display.plot()
plt.show()

