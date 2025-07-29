import os
import struct
import json
import numpy as np
from array import array
from PIL import Image
from torchvision.datasets import MNIST

PATH = "."

def read(dataset):
    if dataset == "training":
        path_img = f"MNIST/raw/train-images-idx3-ubyte"
        path_lbl = f"MNIST/raw/train-labels-idx1-ubyte"
    elif dataset == "testing":
        path_img = f"MNIST/raw/t10k-images-idx3-ubyte"
        path_lbl = f"MNIST/raw/t10k-labels-idx1-ubyte"
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(path_lbl, 'rb') as f_lbl:
        f_lbl.read(8)  # skip magic number and count
        labels = array("b", f_lbl.read())

    with open(path_img, 'rb') as f_img:
        _, _, rows, cols = struct.unpack(">IIII", f_img.read(16))
        images = array("B", f_img.read())

    return labels, images, rows, cols

def write_dataset(labels, data, rows, cols, output_root):
    classes = {i: f"class_{i}" for i in range(10)}
    output_dirs = [os.path.join(output_root, classes[i]) for i in range(10)]

    for dir in output_dirs:
        os.makedirs(dir, exist_ok=True)

    for idx, label in enumerate(labels):
        base_filename = os.path.join(output_dirs[label], f"{idx}.jpg")
        output_filename = get_unique_filename(base_filename)

        print(f'writing {output_filename}')

        data_i = [data[(idx * rows * cols + j * cols):(idx * rows * cols + (j + 1) * cols)] for j in range(rows)]
        data_array = np.asarray(data_i, dtype=np.uint8)
        im = Image.fromarray(data_array)
        im.save(output_filename)

def get_unique_filename(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while True:
        new_path = f"{base} ({i}){ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


def create_reg_dataset():
    ds_path = os.path.join(PATH, 'dataset')
    if not os.path.isdir(ds_path):
        os.makedirs(ds_path)

    img = np.random.randint(0, 50, [100000, 64, 64], dtype=np.uint8)
    square = np.random.randint(100, 200, [100000, 15, 15], dtype=np.uint8)
    coords = np.empty([100000, 2])
    data = {}

    for i in range(img.shape[0]):

        x = np.random.randint(20, 44)
        y = np.random.randint(20, 44)

        img[i, (y - 7):(y + 8), (x - 7):(x + 8)] = square[i]

        coords[i] = [x, y]

        name_img = f'img_{i}.jpeg'
        path_img = os.path.join(ds_path, name_img)

        sample = Image.fromarray(img[i])
        sample.save(path_img)

        data[name_img] = [x, y]

    json_path = os.path.join(ds_path, 'coords.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    MNIST(root=PATH, train=True, download=True)
    MNIST(root=PATH, train=False, download=True)
    output_dir = os.path.join("MNIST", "dataset")
    for dataset in ["training", "testing"]:
        write_dataset(*read(dataset), output_dir)

    create_reg_dataset()

if __name__ == "__main__":
    main()

