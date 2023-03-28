
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dropout, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
import os
import json
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import Callback, CSVLogger
from glob import glob
import pandas as pd



# Check which directory are you in:

# your working directory should be in the main folder where "weights" folder is in.

# print(os.getcwd())

# %cd /content/drive/MyDrive/GDP 

""" 
folder structure:

GDP
  test
  train
  weights

"""

### FUNCTIONS

import time
from tensorflow.keras.callbacks import Callback, CSVLogger
def compute_test(model, X_test_reshaped, y_test):
    # Make predictions on the test data
    y_pred = model.predict(X_test_reshaped)

    # Convert the predicted probabilities to class labels
    y_pred_classes = y_pred.argmax(axis=-1)

    # Compute test accuracy
    correct=0
    for i in range(len(y_pred_classes)):
        if y_pred_classes[i]==y_test[i]:
            correct+=1
    test_accuracy=correct/len(y_pred_classes)

    return test_accuracy, y_pred_classes

class SaveWeightsJSON(Callback):
    def __init__(self, X_test_reshaped, y_test, param_idx, sup_hyper_param_list, model_type, num_frames, num_features, weights_path):
        super(SaveWeightsJSON, self).__init__()
        self.start_time = time.time()
        self.model_name=f"model{param_idx}"
        self.weights_path = weights_path
        self.X_test = X_test_reshaped
        self.y_test = y_test
        
        self.json_data={
                "model_name": self.model_name,
                "model_type": model_type,
                "num_frames": num_frames,
                "num_features": num_features, 
                "num_layers": sup_hyper_param_list[0],
                "dropout_rate": sup_hyper_param_list[1],
                "optimizer": sup_hyper_param_list[2],
                "batch_size": sup_hyper_param_list[3],
                "learning_rate": sup_hyper_param_list[4],
                "best_epoch": 0,
                "best_train_acc": 0.0,
                "best_test_acc": 0.0,
                "training_time": 0.0
        }

        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.best_train_accuracy = 0.0
        self.best_test_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        test_acc, _ = compute_test(self.model, self.X_test, self.y_test)

        if val_loss <= self.best_val_loss:
            if test_acc > self.best_test_accuracy:
                self.best_val_loss = val_loss

                self.best_epoch = epoch + 1
                self.best_train_accuracy = logs.get('accuracy')
                self.best_test_accuracy = test_acc
                weights_file_path = os.path.join(self.weights_path, f"{self.model_name}.h5")
                self.model.save_weights(weights_file_path, overwrite=True)
                                
    def on_train_end(self, logs=None):
        self.json_data['best_epoch'] = self.best_epoch
        self.json_data['best_train_acc'] = self.best_train_accuracy
        self.json_data['best_test_acc'] = self.best_test_accuracy
        json_file_path = os.path.join(self.weights_path, f"{self.model_name}.json")
        self.json_data['training_time'] = time.time() - self.start_time
        with open(json_file_path, 'w') as f:
            json.dump(self.json_data, f, indent=2)

# Imports

import os
import json
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dropout, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD

from sklearn import metrics
import matplotlib.pyplot as plt

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

def opt_function(optimizer,learning_rate):

  if optimizer == 'adam':
    opt = Adam(learning_rate=learning_rate)
  else:
    opt = SGD(learning_rate=learning_rate)

  return opt

def compile_model(sup_hyper_param_list,lstm_units,model_type, num_frame, num_features):

  num_kpsORjnts=int(num_features/2)
  num_layer= sup_hyper_param_list[0]
  dropout_rate= sup_hyper_param_list[1]
  optimizer= sup_hyper_param_list[2]
  #batch_size= sup_hyper_param_list[3]
  learning_rate= sup_hyper_param_list[4]

  opt = opt_function(optimizer,learning_rate)

  num_layer_list = list(range(0,num_layer))
  num_layer_list.reverse()


# Define paramters used in sup_hyper_param_list

  if model_type == 'LSTM':

    model = Sequential ()
    model.add(LSTM(units = (lstm_units)*(2**num_layer), return_sequences=True, input_shape=(num_frame,num_features)))
    model.add(Dropout(dropout_rate))

    for s in num_layer_list:

      if s == 0:
        model.add(LSTM(units = (lstm_units)*(2**s), return_sequences=False))
      else:

        model.add(LSTM(units = (lstm_units)*(2**s), return_sequences=True))
      model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

  elif model_type == 'CNN_LSTM':
    

    model = Sequential ()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(num_frame,num_kpsORjnts,2)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Reshape((model.output_shape[1],model.output_shape[2]*model.output_shape[3])))
    model.add(LSTM(units = (lstm_units)*(2**num_layer), return_sequences=True))
    model.add(Dropout(dropout_rate))

    for s in num_layer_list:
      
      if s == 0:
        model.add(LSTM(units = (lstm_units)*(2**s), return_sequences=False))
      else:

        model.add(LSTM(units = (lstm_units)*(2**s), return_sequences=True))
      model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

  elif model_type == 'BiLSTM':

    model = Sequential ()
    model.add(Bidirectional(LSTM(units = (lstm_units)*(2**num_layer),return_sequences=True), input_shape=(num_frame,num_features)))
    model.add(Dropout(dropout_rate))

    for s in num_layer_list:

      if s == 0:
        model.add(Bidirectional(LSTM(units = (lstm_units)*(2**s),return_sequences=False)))
      else:

        model.add(Bidirectional(LSTM(units = (lstm_units)*(2**s),return_sequences=True)))
      model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
  
  elif model_type == 'CNN_BiLSTM':
  
    model = Sequential ()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(num_frame,num_kpsORjnts,2)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Reshape((model.output_shape[1],model.output_shape[2]*model.output_shape[3])))
    model.add(Bidirectional(LSTM(units = (lstm_units)*(2**num_layer),return_sequences=True), input_shape=(num_frame,num_features)))
    model.add(Dropout(dropout_rate))
    for s in num_layer_list:

      if s == 0:
        model.add(Bidirectional(LSTM(units = (lstm_units)*(2**s),return_sequences=False)))
      else:

        model.add(Bidirectional(LSTM(units = (lstm_units)*(2**s),return_sequences=True)))
      model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
  else:
    print('Invalid model name. You can only use these models: "CNN","CNN_LSTM","BiLSTM","CNN_BiLSTM"')


def run_models(hyper_param_list,epochs,lstm_units,split_train_path,split_test_path,weights_path,model_type):

    print("Number of configurations to run: ", len(hyper_param_list))
    X_train, y_train = load_data(split_train_path)
    num_json,num_frames,num_features = X_train.shape 

    X_test, y_test = load_data(split_test_path)
    num_json_test = X_test.shape[0]

    num_classes = 3
    y_train_cat=to_categorical(y_train,num_classes)
    y_test_cat=to_categorical(y_test,num_classes)

    num_kpsORjnts=int(num_features/2)

    if model_type == 'CNN_LSTM' or model_type == 'CNN_BiLSTM':

      X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_kpsORjnts, 2))
      X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_kpsORjnts, 2))

    else:
      pass

    for param_idx, sup_hyper_param_list in enumerate(hyper_param_list): 
      print(sup_hyper_param_list)
      save_weights_json = SaveWeightsJSON(X_test_reshaped=X_test, y_test=y_test, param_idx=param_idx,sup_hyper_param_list=sup_hyper_param_list,
                                        model_type=model_type, num_frames=num_frames, num_features=num_features, weights_path=weights_path)
      
      log_file_path = os.path.join(weights_path, f"model{param_idx}.log")
      save_log = CSVLogger(log_file_path)

      model = compile_model(sup_hyper_param_list=sup_hyper_param_list,lstm_units=lstm_units,
                            model_type=model_type, num_frame = num_frames, num_features = num_features)
      
      model_fit = model.fit(X_train, y_train_cat, epochs=epochs, validation_split = 0.2, batch_size=sup_hyper_param_list[3],
                            callbacks=[save_weights_json,save_log])
      

### IMPLEMENTATION

""" Running the desired model with several parameters using run_models():

hyper_param_list  = DO NOT CHANGE. Keep as it is in the function call.

epochs = Number of epochs you would like to try for each of the model combination

lstm_units = LSTM layer units (recommendation: do not increase it too much, start with 16,32, 64)

split_train_path = Path to train data

split_train_path = Path to test data

weights_path = Path to weights folder which all files regarding to model will be created

model_type = Which model is you want to try? 

    options for model_type are:
              LSTM
              CNN_LSTM
              BiLSTM
              CNN_BiLSTM

"""    

# You CAN change param_grid depending on which parameters you would like to try in the model:

param_grid = {
              'dropout_rate': [0.1],
              'batch_size': [64],
              'num_layer': [1,2],
              'optimizer': ['adam'],
              'learning_rate': [0.01]
              }

# Do NOT change hyper_param_list!

hyper_param_list=[(num_layer, dropout_rate, optimizer, batch_size, learning_rate)
                  for num_layer in param_grid['num_layer']
                  for dropout_rate in param_grid['dropout_rate']
                  for optimizer in param_grid['optimizer']
                  for batch_size in param_grid['batch_size']
                  for learning_rate in param_grid['learning_rate']
                  ]


# Specify all three paths:

split_train_path = r"C:\Users\Ivan Lopez\Desktop\GDP\split_seqs\run1_a+p_dist_1_0.5\train"
split_test_path = r"C:\Users\Ivan Lopez\Desktop\GDP\split_seqs\run1_a+p_dist_1_0.5\test"
weights_path = r"C:\Users\Ivan Lopez\Desktop\GDP\weights"

# --- Irem - Colab Version  ---
#split_train_path="/content/drive/MyDrive/GDP/train"
#split_test_path="/content/drive/MyDrive/GDP/test"
#weights_path="/content/drive/MyDrive/GDP/weights"


# Run a desired model using the belove line:

run_models(hyper_param_list=hyper_param_list, epochs=2, lstm_units=16,
            split_train_path=split_train_path,split_test_path=split_test_path,
            weights_path=weights_path, model_type="LSTM")
