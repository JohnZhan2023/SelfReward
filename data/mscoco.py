'''
change the mscoco to a dataset
'''
import os
import json
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
class MSCOCO(Dataset):
    def __init__(self, root, ann_file, transform=None, max_samples=20000):
        self.root = root
        self.coco = COCO(ann_file)
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = self.ids[:max_samples]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB') # PIL image
        # turn the image into a tensor
        img = np.array(img)
        if self.transform:
            img = self.transform(img)
        return img, anns, img_info

# Collate function to resize images to the same size and stack them into a batch
def collate_fn(batch):
    imgs, anns, img_info = zip(*batch)
    
    # Resize images to the same size, for example (256, 256)
    resize_transform = transforms.Resize((256, 256))  # Resize to (256, 256)
    
    imgs_resized = []
    for img in imgs:
        img_resized = Image.fromarray(img)  # Convert numpy array back to PIL image
        img_resized = resize_transform(img_resized)
        img_resized = np.array(img_resized)  # Convert back to numpy array if necessary
        imgs_resized.append(torch.tensor(img_resized).permute(2, 0, 1))  # Convert to tensor (C, H, W)
    
    # Stack the images into a single tensor
    imgs_resized = torch.stack(imgs_resized)  # Shape: (batch_size, C, H, W)
    
    return imgs_resized, anns, img_info


def prompt_merge(anns):
    merged_prompt = f'''
    Here are the five captions from different perspectives of the same scene:
    {anns[0]['caption']}
    {anns[1]['caption']}
    {anns[2]['caption']}
    {anns[3]['caption']}
    {anns[4]['caption']}
    Please draw a picture based on these five captions.
    '''
    return merged_prompt


if __name__ == '__main__':
    root = '/mnt/disk5/zhanjh/mscoco/train2017'
    ann_file = '/mnt/disk5/zhanjh/mscoco/annotations/captions_train2017.json'
    mscoco = MSCOCO(root, ann_file)
    print("the length of mscoco caption is: ", len(mscoco))
    
    img, anns, img_info = mscoco[1]
    print(img.shape)  # 查看img的形状

    # 如果 img 是 (H, W, C) 形式的 numpy 数组，直接显示图像
    plt.imshow(img)  # 显示图像
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 渲染图像
    
    # 输出图像的相关信息
    print("Annotations: ", anns)
    print("Image info: ", img_info)

    # 打印 caption 示例
    i = 0
    for ann in anns:
        caption = ann['caption']
        print(f"Caption {i}: {caption}")
        i += 1