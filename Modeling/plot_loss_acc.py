import csv
import json

# Specify the path of the json and log file
log_path = r'path/to/model/log/file'
json_path = r'path/to/model/json/file'

# Initialize empty lists to store the header and data rows of the log
header = []
data = []

# Open the log file and read the data using csv.reader
with open(log_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if not header:
            # Store the header row
            header = row
        else:
            # Store the data rows
            data.append(row)

# Convert the data into a dictionary where the keys are the column names
log_data = {header[i]: [float(row[i]) for row in data] for i in range(len(header))}


with open(json_path) as f:
    data = json.load(f)
best_epoch=data['best_epoch']


import matplotlib.pyplot as plt

# Extract the train loss and val_loss data
train_loss = log_data['loss']
val_loss = log_data['val_loss']
train_acc = log_data['accuracy']
val_acc = log_data['val_accuracy']

# Plot the train loss and val_loss over epochs

fig = plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

epochs = range(1, len(train_loss) + 1)
ax1.plot(epochs, train_loss, 'b', label='Training loss')
ax1.plot(epochs, val_loss, 'r', label='Validation loss')
ax1.plot(best_epoch, val_loss[best_epoch-1], 'o', label='Best Test Accuracy')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()



ax2.plot(epochs, train_acc, 'b', label='Training Accuracy')
ax2.plot(epochs, val_acc, 'r', label='Validation Accuracy')
ax2.plot(best_epoch, val_acc[best_epoch-1], 'o', label='Best Test Accuracy')
ax2.set_title('Training and validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()