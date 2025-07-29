import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils import transform

class MNISTDataset(Dataset):
    def __init__(self, path):
        self.len_dataset = 0
        self.data_list = []
        
        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.classes_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                self.idx_to_classes = {v: k for k, v in self.classes_to_idx.items()}
                continue

            cls = path_dir.split(os.sep)[-1]

            for file_name in file_list:
                file_path = os.path.join(path_dir, file_name)
                self.data_list.append((file_path, self.classes_to_idx[cls]))

            self.len_dataset += len(file_list)
        
        self.shape = Image.open(self.data_list[0][0]).size
    
    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, i):
        file_path, y = self.data_list[i]
        x = transform(Image.open(file_path))
        return x, y
    
    def view(self, i):
        img = Image.open(self.data_list[i][0])
        cls = self.idx_to_classes[self.data_list[i][1]]
        return img, cls
    
class REGRDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.list_name_files = os.listdir(self.path)

        if 'coords.json' in self.list_name_files:
            self.list_name_files.remove('coords.json')
            with open(os.path.join(self.path, 'coords.json'), 'r') as f:
                self.dict_coords = json.load(f)

        self.len_dataset = len(self.list_name_files)
        self.shape = Image.open(os.path.join(path, self.list_name_files[0])).size
    
    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, i):
        file_name = self.list_name_files[i]
        image_path = os.path.join(self.path, file_name)
        x = transform(Image.open(image_path))
        y = torch.tensor(self.dict_coords[file_name], dtype=torch.float32)
        return x, y
    
    def view(self, i):
        file_name = self.list_name_files[i]
        img = Image.open(os.path.join(self.path, file_name))
        coord = self.dict_coords[file_name]
        return img, coord