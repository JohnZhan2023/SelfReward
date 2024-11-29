import os
import json
from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class MSCOCO_WinLoss(Dataset):
    def __init__(self, root, neg_root, ann_file, transform=None, max_samples=20000, mode='train'):
        """
        mode: 'train' for training, 'val' for validation
        """
        self.root = root
        self.neg_root = neg_root
        self.coco = COCO(ann_file)
        self.transform = transform
        
        # Load all image ids and split them into train/validation sets
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = self.ids[:max_samples]  # Limit the dataset size to max_samples
        
        # Split into train and validation sets (e.g., 80% train, 20% val)
        train_size = int(0.9 * len(self.ids))  # Use 90% for training
        val_size = len(self.ids) - train_size  # The rest for validation
        
        # Split the dataset
        if mode == 'train':
            self.ids = self.ids[:train_size]
        elif mode == 'val':
            self.ids = self.ids[train_size:]
        
    def __len__(self):
        return len(self.ids) * 2  # Each image pair will contribute two samples: one for real and one for fake

    def __getitem__(self, idx):
        # Find the corresponding image index (original image and fake image)
        img_idx = self.ids[idx // 2]  # Use integer division to pair up images
        is_real = idx % 2 == 0  # Even indices correspond to real images, odd indices to fake images
        
        # Load annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_idx)
        anns = self.coco.loadAnns(ann_ids)
        
        img_info = self.coco.loadImgs(img_idx)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')  # PIL image
        
        if is_real:
            label = torch.tensor([1, 0])  # Label 1 for real image
        else:
            fake_img = Image.open(os.path.join(self.neg_root, img_info['file_name'])).convert('RGB')
            img = fake_img  # For fake images, replace the real image with the fake one
            label = torch.tensor([0, 1])  # Label 0 for fake image
        
        # Apply transformation (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)
        
        return img, label, anns, img_info

# Collate function to resize images to the same size and stack them into a batch
def collate_fn(batch):
    imgs, labels, anns, img_info = zip(*batch)
    
    # Resize images to the same size, for example (256, 256)
    resize_transform = transforms.Resize((256, 256))  # Resize to (256, 256)
    
    # Resize images (already converted to tensor after resize)
    imgs_resized = [resize_transform(img) for img in imgs]

    # Stack the images into a single tensor (B, C, H, W)
    # No need for ToTensor() since imgs_resized is already a tensor
    imgs_resized = torch.stack(imgs_resized)
    # Stack the labels into a single tensor (B, 2), since the labels are 2D
    labels = torch.stack(labels)  # Convert labels list of 2D tensors into a single tensor
    
    
    return imgs_resized, labels, anns, img_info


if __name__ == '__main__':
    root = '/mnt/disk5/zhanjh/mscoco/train2017'
    neg_root = '/mnt/disk5/zhanjh/mscoco/negative_sample'
    ann_file = '/mnt/disk5/zhanjh/mscoco/annotations/captions_train2017.json'
    
    transform = transforms.Compose([  # Transformations
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.ToTensor(),  # Convert to tensor (C, H, W)
    ])
    
    # Initialize the dataset for training and validation
    train_dataset = MSCOCO_WinLoss(root, neg_root, ann_file, transform=transform, mode='train')
    val_dataset = MSCOCO_WinLoss(root, neg_root, ann_file, transform=transform, mode='val')

    print("Length of training dataset: ", len(train_dataset))
    print("Length of validation dataset: ", len(val_dataset))
    
    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Example for iterating through the dataset
    for img, label, anns, img_info in train_loader:
        print("Train batch - Image shape:", img.shape, "Label shape:", label.shape)
        break
    
    # Evaluate on the validation set
    for img, label, anns, img_info in val_loader:
        print("Validation batch - Image shape:", img.shape, "Label shape:", label.shape)
        break
        
        # Add model evaluation logic here
        # predictions = model(img)
        # Evaluate the model on the validation data
