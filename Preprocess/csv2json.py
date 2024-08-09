# Author 
# Rahul Kumar (Northeastern University)

import os
import pandas as pd
import json

def CSVtoJSON(csv_file_path,image_folder, json_file):
    """
    Convert CSV data to JSON format and associate each record with corresponding images from specified folders.

    This function reads medical imaging data from a CSV file, locates the associated images in the specified
    folders, and writes this information into JSON files. Each JSON file corresponds to a subset of the data
    (e.g., training or testing set).

    Args:
    csv_file_path (str): Path to the CSV file containing the data.
    image_folder (list[str]): List of paths to the folders containing the images. Each folder corresponds to a
                              data subset.
    json_file (list[str]): List of paths where the JSON files will be saved. Each path corresponds to a
                           data subset.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    for i in range(2):
        json_data = []

        for index, row in df.iterrows():
            uid = row['uid']
            findings = row['findings']
            
            # Find the image file that starts with the UID
            image_file_name = []
            for file_name in os.listdir(image_folder[i]):
                if file_name.startswith(f"{uid}_"):
                    image_file_name.append(file_name)
                    
            
            for name in image_file_name:
                img_path = os.path.join(image_folder[i], name)
                
                json_data.append({
                    "uid": uid,
                    "findings": findings,
                    "img_path": img_path
                })


        # Convert the list to JSON format
        with open(json_file[i], 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON data has been written to {json_file[i]}")

if __name__ == "__main__":
    csv_file_path = '../Dataset/Open-i/indiana_reports_processed.csv'
    train_image_folder_path = '../Dataset/Open-i/train'
    train_json_file_path = '../Dataset/Open-i/train.json'
    test_image_folder_path = '../Dataset/Open-i/test'
    test_json_file_path = '../Dataset/Open-i/test.json'

    image_folder = [train_image_folder_path, test_image_folder_path]
    json_file = [train_json_file_path, test_json_file_path]

    CSVtoJSON(csv_file_path,image_folder,json_file)