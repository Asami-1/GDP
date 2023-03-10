
print("aaaaa")
import os
#path = "Output_vitposea"

import os

def rename_dirs(path):
    # go through Agressive et Peace
    for subdir in ['Agressive', 'Peace']:
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            # go through les sous-dossiers
            for subsubdir_name in os.listdir(subdir_path):
                subsubdir_path = os.path.join(subdir_path, subsubdir_name)
                if os.path.isdir(subsubdir_path):
                    # Renommer le sous-dossier en utilisant le nouveau nom
                    new_subsubdir_name = f"{subdir}_{subsubdir_name}"
                    new_subsubdir_path = os.path.join(subdir_path, new_subsubdir_name)
                    os.rename(subsubdir_path, new_subsubdir_path)


rename_dirs("TEST") 

        

#doesnt work  because folders (and not files) inside Agressive and Peace
    # def split_data(input_path, output_path, split_ratio=0.75):
    #     # Creatre subpath
    #     train_path = os.path.join(output_path, "Train")
    #     test_path = os.path.join(output_path, "Test")
    #     os.makedirs(train_path, exist_ok=True)
    #     os.makedirs(test_path, exist_ok=True)
    # Go through Agressive and Peace 
    # for subdir in ['Agressive', 'Peace']:
    #     subdir_path = os.path.join(input_path, subdir)
    #     if os.path.isdir(subdir_path):
    #         # Sélectionner les fichiers dans le sous-dossier
    #         files = [file_name for file_name in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, file_name))]
    #         # Shuffle the files, not really necessary 
    #         random.shuffle(files)
    #         # Separate 
    #         num_files = len(files)
    #         num_train = int(num_files * split_ratio)
    #         train_files = files[:num_train]
    #         test_files = files[num_train:]
    #         # Copier les fichiers dans les dossiers Train et Test
    #         for file_name in train_files:
    #             src_path = os.path.join(subdir_path, file_name)
    #             dst_path = os.path.join(train_path, subdir, file_name)
    #             os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #             shutil.copy(src_path, dst_path)
    #         for file_name in test_files:
    #             src_path = os.path.join(subdir_path, file_name)
    #             dst_path = os.path.join(test_path, subdir, file_name)
    #             os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #             shutil.copy(src_path, dst_path)
    #         print(f"{num_files} fichiers dans le sous-dossier {subdir}: {num_train} dans Train, {num_files - num_train} dans Test.")

    # # Compter le nombre total de fichiers dans les sous-dossiers Train
    # num_train_aggressive = len(os.listdir(os.path.join(train_path, "Agressive")))
    # num_train_peace = len(os.listdir(os.path.join(train_path, "Peace")))
    # total_train_files = num_train_aggressive + num_train_peace

# import os
# import shutil
# import random

# def split_data(input_path, output_path, split_ratio=0.75):
#     # Créer le dossier Data avec les sous-dossiers Train et Test
#     train_path = os.path.join(output_path, "Train")
#     test_path = os.path.join(output_path, "Test")
#     os.makedirs(train_path, exist_ok=True)
#     os.makedirs(test_path, exist_ok=True)

#     # Parcourir les sous-dossiers Agressive et Peace
#     for subdir in ['Agressive', 'Peace']:
#         subdir_path = os.path.join(input_path, subdir)
#         if os.path.isdir(subdir_path):
#             # Parcourir  tous les fichiers dans le sous-dossier
#             files = []
#             for dirpath, dirnames, filenames in os.walk(subdir_path):
#                 for filename in filenames:
#                     files.append(os.path.join(dirpath, filename))
#             # Mélanger les fichiers aléatoirement
#             random.shuffle(files)
#             # Séparer les fichiers en train et test
#             num_files = len(files)
#             num_train = int(num_files * split_ratio)
#             train_files = files[:num_train]
#             test_files = files[num_train:]
#             # Copier les fichiers dans les dossiers Train et Test
#             for file_path in train_files:
#                 src_path = file_path
#                 dst_path = os.path.join(train_path, subdir, os.path.relpath(file_path, subdir_path))
#                 os.makedirs(os.path.dirname(dst_path), exist_ok=True)
#                 shutil.copy(src_path, dst_path)
#             for file_path in test_files:
#                 src_path = file_path
#                 dst_path = os.path.join(test_path, subdir, os.path.relpath(file_path, subdir_path))
#                 os.makedirs(os.path.dirname(dst_path), exist_ok=True)
#                 shutil.copy(src_path, dst_path)

#     # Compter le nombre total de fichiers dans les sous-dossiers Train
#     num_train_aggressive = len([name for name in os.listdir(os.path.join(train_path, "Agressive")) if os.path.isfile(os.path.join(train_path, "Agressive", name))])
#     num_train_peace = len([name for name in os.listdir(os.path.join(train_path, "Peace")) if os.path.isfile(os.path.join(train_path, "Peace", name))])
#     total_train_files = num_train_aggressive + num_train_peace


#split_data("TEST"," TEST_OUTPUT")    
