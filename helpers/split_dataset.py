import os
import shutil
from random import shuffle

# Set path to the directory containing the dataset folders
path = "dataset"

# Get a list of the dataset folders
folders = ["test_A", "test_B", "test_C"]

# Create a new folder for the evaluation dataset
os.makedirs("eval_dataset")

# Iterate over the dataset folders
for folder in folders:
    # Get the path to the current folder
    current_folder = os.path.join(path, folder)

    # Get a list of all the images in the folder
    images = os.listdir(current_folder)
    shuffle(images)
    shuffle(images)

    # Calculate the number of images that should be in the evaluation dataset
    eval_size = int(len(images) * 0.2)

    # Create a new folder for the evaluation dataset
    eval_folder = os.path.join("eval_dataset", folder)
    os.makedirs(eval_folder)

    # Move the required number of images to the evaluation dataset folder
    for i in range(eval_size):
        image = images[i]
        src_path = os.path.join(current_folder, image)
        dst_path = os.path.join(eval_folder, image)
        shutil.move(src_path, dst_path)

