import os
import glob
import cv2
import random
from utils import read_output_txt, read_img, create_raw_dataset, check_data_vs_output_quantity
from config import SAVE_IMG_DIR, SAVE_OUTPUT_DIR

# SAVE THE SHUFFLED DATASET


def pair_img_output():
    dataset = create_raw_dataset(tensor=0)
    return dataset


def save_dataset():
    dataset = pair_img_output()
    dataset_size = len(dataset)
    print("Save images and outputs at {}".format(SAVE_IMG_DIR))

    for i in range(0, dataset_size):
        # Save images
        img_file = f"{SAVE_IMG_DIR}/{i+1}-{dataset[i][1]}:{dataset[i][2]:.4f}.png"
        print("saving {}".format(img_file))
        cv2.imwrite(img_file, dataset[i][0])

    # Save their corresponding outputs
    with open(SAVE_OUTPUT_DIR, 'w') as f:
        for i in range(0, dataset_size):
            if i == dataset_size - 1:
                f.write(f"{dataset[i][2]}")
            else:
                f.write(f"{dataset[i][2]}\n")

    print("Complete")


if __name__ == '__main__':
    save_dataset()