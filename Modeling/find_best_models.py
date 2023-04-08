import os
import json
import numpy as np


top_X=10

run_path=r"path/to/run"
seq_duration=['1','2']
type_feature=['dist']
models=['LSTM', 'BiLSTM', 'CNN_LSTM',  'CNN_BiLSTM']

best_X_list = []
best_test_acc = 0.0

for type_f in type_feature:
    for dur in seq_duration:
        folder_name=f"weights_{type_f}_{dur}"
        for model in models:
            weights_path=os.path.join(run_path,folder_name,model)
            weighs_list=os.listdir(weights_path)
            for file in weighs_list:
                if file.endswith('.json'):
                    file_path = os.path.join(weights_path, file)
                    with open(file_path) as f:
                        data = json.load(f)
                    if len(best_X_list)<top_X:
                        best_X_list.append(data)
                    else:  
                        # Find the model with the lowest best_test_acc in the top 10
                        min_index = 0
                        for i in range(1,len(best_X_list)):
                            if best_X_list[i]['best_test_acc'] < best_X_list[min_index]['best_test_acc']:
                                min_index = i
                        # Replace the model with the lowest best_test_acc if the new model has a higher best_test_acc
                        if data['best_test_acc'] > best_X_list[min_index]['best_test_acc']:
                            best_X_list[min_index] = data

sorted_list = sorted(best_X_list, key=lambda x: -x['best_test_acc'])
print('\n')
for idx, model in enumerate(sorted_list):
    if model['num_features']==24:
        type_of='dist'
    else:
        type_of='angl'
    print(idx+1, model['model_type'],'-- Seq param:',int(model['num_frames']/10),type_of, '-- Model', model['model_name'], '-- Train accuracy', model['best_train_acc'], '-- Test accuracy', model['best_test_acc'])
    print('              ', model['num_layers'], model['dropout_rate'], model['optimizer'], model['batch_size'], model['learning_rate'])
print('\n')