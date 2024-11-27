'''
change the mscoco to a dataset
'''
import os
import json
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset
class MSCOCO(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, anns, img_info
