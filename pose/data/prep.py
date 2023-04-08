



import os
"""
def rename_subfolders(path, prefix):
    for i, foldername in enumerate(sorted(os.listdir(path), key=lambda x: int(x) if not x.startswith(".") else float("inf"))):
        oldname = os.path.join(path, foldername)
        newname = os.path.join(path, f"{prefix}{i+1}")
        os.rename(oldname, newname)
        print(f"Renamed {oldname} to {newname}")



path = r"result_conflict_cam2"
prefix = "Agressive_cam2_"
rename_subfolders(path, prefix)
"""




"""
import os
import shutil

def merge_folders(source_folder, destination_folder):

    
    # On récupère les noms des sous-dossiers de source_folder
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    
    # Pour chaque sous-dossier, on vérifie s'il existe déjà dans destination_folder
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        destination_subfolder = os.path.join(destination_folder, subfolder_name)
        
        # Si le sous-dossier n'existe pas encore dans destination_folder, on le copie simplement
        if not os.path.exists(destination_subfolder):
            shutil.copytree(subfolder, destination_subfolder)
        
        # Si le sous-dossier existe déjà dans destination_folder, on le renomme en ajoutant un numéro
        else:
            i = 2
            while os.path.exists(destination_subfolder):
                destination_subfolder = os.path.join(destination_folder, subfolder_name + f" ({i})")
                i += 1
            shutil.copytree(subfolder, destination_subfolder)

# Exemple d'utilisation de la fonction merge_folders
source_folder = "Peace_test"
destination_folder = "Agressive_test"
merge_folders(source_folder, destination_folder)
"""



"""
import os

def rename_subfolders(folder_path, prefix):

    
    # On récupère les noms des sous-dossiers du dossier
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    # On renomme chaque sous-dossier avec le nouveau nom
    for i, subfolder in enumerate(subfolders):
        new_name = f"{prefix}_{i+1}"
        os.rename(subfolder, os.path.join(folder_path, new_name))

# Exemple d'utilisation de la fonction rename_subfolders
folder_path = "result_peace_cam1"
prefix = "Peace"
rename_subfolders(folder_path, prefix)
"""
 



import os
import random
import shutil

def split_folders(folder_path, train_ratio=0.75):
    
    #Split subfolders in a folder into train and test subfolders.
    
    #Parameters:
        #folder_path (str): The path to the folder to be split.
        #train_ratio (float): The ratio of subfolders to be placed in the train folder.
        
    #Returns:
    #    None
    
    # Create the train and test folders
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Get the subfolder names
    subfolder_names = os.listdir(folder_path)
    
    # Shuffle the subfolder names
    random.shuffle(subfolder_names)
    
    # Split the subfolder names into train and test sets
    num_train = int(train_ratio * len(subfolder_names))
    train_subfolder_names = subfolder_names[:num_train]
    test_subfolder_names = subfolder_names[num_train:]
    
    # Move the train subfolders
    for subfolder_name in train_subfolder_names:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        train_subfolder_path = os.path.join(train_folder, subfolder_name)
        os.makedirs(train_subfolder_path, exist_ok=True)
        
        # Iterate through the files in the subfolder
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.json'):
                train_file_path = os.path.join(train_subfolder_path, file_name)
                shutil.copy(file_path, train_file_path)
    
    # Move the test subfolders
    for subfolder_name in test_subfolder_names:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        test_subfolder_path = os.path.join(test_folder, subfolder_name)
        os.makedirs(test_subfolder_path, exist_ok=True)
        
        # Iterate through the files in the subfolder
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.json'):
                test_file_path = os.path.join(test_subfolder_path, file_name)
                shutil.copy(file_path, test_file_path)

split_folders("Agressive", train_ratio=0.75)
