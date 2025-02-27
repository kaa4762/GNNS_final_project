import kornia
import zipfile
import os
import pandas as pd
import numpy as np
import cv2
from io import BytesIO

# Step 1: Extract images from ZIP into a list
def extract_images(zip_path):
    image_list = []  # Store (filename, img_array)
    
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for file_name in archive.namelist():
            if file_name.lower().endswith('.png'):  # Only process PNG images
                with archive.open(file_name) as image_file:
                    print('processing ', file_name)
                    img_array = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                    image_list.append((file_name, img_array))    
    return image_list

# Step 2: Read CSV and match files
def load_csv_data(csv_path, image_list):
    df = pd.read_csv(csv_path)
    
    # Convert filenames from CSV to match ZIP contents
    df["ImageID"] = df["ImageID"].astype(str)

    # Filter only rows that match our image filenames
    img_filenames = set(f[0] for f in image_list)
    df_filtered = df[df["ImageID"].isin(img_filenames)]

    return df_filtered

# Step 3: Combine into final dataset
def create_dataset(image_list, df_filtered):
    dataset = []
    
    for filename, img_array in image_list:
        row = df_filtered[df_filtered["ImageID"] == filename]

        if not row.empty:
            orientation = row["Projection"].values[0]  # AP, PA, L, etc.
            labels = row["Labels"].values[0]           # Diagnostic findings
        else:
            orientation = None
            labels = None
        
        dataset.append((filename, img_array, orientation, labels))
    
    return dataset

def save_as_pickle(dataset):
    import pickle
    with open("./data/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    """ load the first part of the PADCHEST dataset in 0.zip and dump in pickle file;
    dataset has form (filename, img_array, orientation, labels):
    e.g. ('135803415504923515076821959678074435083_fzis7d.png', array([[ 65,  64,  63, ...,   0,   0,   0],
       [ 65,  64,  63, ...,   0,   0,   0],
       [ 65,  64,  63, ...,   0,   0,   0],
       ...,
       [187, 190, 191, ...,   0,   0,   0],
       [189, 191, 192, ...,   0,   0,   0],
       [193,   0,   0, ...,   0,   0,   0]], dtype=uint8), 'L', "['pulmonary fibrosis', 'chronic changes', 'kyphosis', 'pseudonodule', 'ground glass pattern']")"""
    

    # Define dataset path
    zip_path = './data/0.zip'
    #define labels path
    csv_path = './data/PADCHEST_labels.csv'
    # Run the functions
    image_list = extract_images(zip_path)
    df_filtered = load_csv_data(csv_path, image_list)
    dataset = create_dataset(image_list, df_filtered)

    save_as_pickle(dataset)

    # Print some dataset examples
    for i in range(5):
        print(dataset[i])