import os
import json
import shutil



def create_dataset_with_duration(data_dir, output_dir, video_duration):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each subfolder in the data directory
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            output_subfolder_path = os.path.join(output_dir, subfolder)
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            # Loop through each sub-subfolder in the subfolder
            for sub_subfolder in os.listdir(subfolder_path):
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
                if os.path.isdir(sub_subfolder_path):
                    output_sub_subfolder_path = os.path.join(output_subfolder_path, sub_subfolder)
                    if not os.path.exists(output_sub_subfolder_path):
                        os.makedirs(output_sub_subfolder_path)
                    

                    # Loop through each sub-subfolder in the subfolder
                    for sub_sub_subfolder in os.listdir(sub_subfolder_path):
                        sub_sub_subfolder_path = os.path.join(sub_subfolder_path, sub_sub_subfolder)
                        if os.path.isdir(sub_sub_subfolder_path):
                            output_sub_sub_subfolder_path = os.path.join(output_sub_subfolder_path, sub_sub_subfolder)
                            if not os.path.exists(output_sub_sub_subfolder_path):
                                os.makedirs(output_sub_sub_subfolder_path)

                            # Loop through each Json file in the sub-subfolder
                            json_files = [f for f in os.listdir(sub_sub_subfolder_path)]
                            num_json_files = min(int(video_duration*30), len(json_files))
                            print(num_json_files)
                            for json_file in sorted(json_files)[:num_json_files]:
                                # Load the Json file
                                json_path = os.path.join(sub_sub_subfolder_path, json_file)

                                with open(json_path) as f:
                                    data = json.load(f)
                                # Copy the Json file to the output sub-subfolder
                                output_json_path = os.path.join(output_sub_sub_subfolder_path, json_file)
                                shutil.copy2(json_path, output_json_path)


create_dataset_with_duration("data_full", "data_time_fixed",1)
