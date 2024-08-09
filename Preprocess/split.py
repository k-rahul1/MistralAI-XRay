# Author 
# Rahul Kumar (Northeastern University)

import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_folders(base_path):
    """
    Create subdirectories for training, testing, and validation sets within a given base directory.

    Args:
    base_path (str): The path to the base directory where the subdirectories will be created.

    Creates:
    Three folders named 'train', 'test', and 'val' inside the specified base_path. Each folder is created if it does not already exist.
    """
    os.makedirs(os.path.join(base_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val'), exist_ok=True)

def split_data(image_folder, base_path, train_size=0.8, test_size=0.1, val_size=0.1):
    """
    Split a set of images into training, testing, and validation datasets and copy them into respective folders.

    Args:
    image_folder (str): The directory containing the images to be split.
    base_path (str): The base directory where the 'train', 'test', and 'val' folders exist or will be created.
    train_size (float): The proportion of the dataset to include in the train split (default is 0.8).
    test_size (float): The proportion of the dataset to include in the test split (default is 0.1).
    val_size (float): The proportion of the dataset to include in the validation split (default is 0.1).

    Outputs:
    Copies the images into the appropriate subdirectories under base_path according to the splits specified.
    Prints the number of images in each split.
    """
    images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    print(f'Total images: {len(images)}')
    train, temp = train_test_split(images, train_size=train_size)
    test, val = train_test_split(temp, test_size=test_size / (test_size + val_size))

    print(f'Number of training images: {len(train)}')
    print(f'Number of testing images: {len(test)}')
    print(f'Number of validation images: {len(val)}')

    for dataset, folder in zip([train, test, val], ['train', 'test', 'val']):
        print(f'Copying images to {folder} folder...')
        for image in tqdm(dataset, desc=f'{folder} images'):
            shutil.copy(os.path.join(image_folder, image), os.path.join(base_path, folder, image))

if __name__ == "__main__":
    image_folder = '../Dataset/Open-i/images/images_normalized/'
    base_path = '../Dataset/Open-i'

    create_folders(base_path)
    split_data(image_folder, base_path)
