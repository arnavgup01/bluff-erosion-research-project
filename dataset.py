import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class BluffDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        self.transform_image = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        self.transform_mask = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        matching_masks = [f for f in os.listdir(self.mask_dir) if f.startswith(base_name + "_masked")]
        if len(matching_masks) == 0:
            raise FileNotFoundError(f"No mask found for image: {img_name}")
        if len(matching_masks) > 1:
            raise ValueError(f"Multiple masks found for image: {img_name} -> {matching_masks}")

        mask_name = matching_masks[0]
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        mask = (mask > 0).float()

        return image, mask