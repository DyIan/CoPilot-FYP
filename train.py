import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import torch
import numpy as np
from enet import ENet

from torch.utils.data import DataLoader




class SegmentationDataset(Dataset):
    def __init__(self, image_list_file, img_dir, mask_dir, transform=None):
        with open(image_list_file, 'r') as f:
            self.files = [line.strip() for line in f.readlines()]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

class SegmentationTransform:
    def __init__(self, resize=(256, 256), train=True):
        self.resize = resize
        self.train = train

    def __call__(self, image, mask):
        image = TF.resize(image, self.resize)
        mask = TF.resize(mask, self.resize, interpolation=Image.NEAREST)

        if self.train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Optional: brightness/contrast jitter
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = np.array(mask, dtype=np.uint8)
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask

def pixel_accuracy(output, mask):
    # output shape: [batch, num_classes, H, W]
    # mask shape: [batch, H, W]
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        valid = torch.ones_like(mask, dtype=torch.bool)
        correct = (preds == mask) & valid
        accuracy = correct.sum().float() / valid.sum().float()
    return accuracy.item()


num_classes = 3
model = ENet(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 

import torch.optim as optim
import torch.nn as nn

# Lane markings show up less so need to weight them to stop class imbalance
weights = torch.tensor([1.0, 1.0, 4.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)



train_dataset = SegmentationDataset("data/split/train.txt", "data/rgb", "data/mask", transform=SegmentationTransform(train=True))
val_dataset = SegmentationDataset("data/split/val.txt", "data/rgb", "data/mask", transform=SegmentationTransform(train=False))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    running_acc = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += pixel_accuracy(outputs, masks) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    running_acc = 0

    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            running_acc += pixel_accuracy(outputs, masks) * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            # Iou
            for cls in range(num_classes):
                pred_inds = (preds == cls)
                target_inds = (masks == cls)

                total_intersection[cls] += (pred_inds & target_inds).sum().float()
                total_union[cls] += (pred_inds | target_inds).sum().float()

    iou_classes = {}
    for cls in range(num_classes):
        if total_union[cls] == 0:  # No class present
            iou_classes[cls] = float('nan')
        else:
            iou_classes[cls] = (total_intersection[cls] / total_union[cls]).item()

    valid_ious = [v for v in iou_classes.values() if not np.isnan(v)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc, iou_classes, mean_iou






def train():
    num_epochs = 100
    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_iou, val_miou = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Train Loss: {train_loss:.4f} — Train Acc: {train_acc:.4f} — "
              f"Val Loss: {val_loss:.4f} — Val Acc: {val_acc:.4f} - "
              f"mIoU: {val_miou:.4f}")

        print(f"IoU — Background: {val_iou[0]:.3f}, "
              f"Road: {val_iou[1]:.3f}, "
              f"Lane: {val_iou[2]:.3f}"
)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_enet.pth')
            print("Saved best model.")


if __name__ == "__main__":
    train()
