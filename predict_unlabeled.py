import os
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import segmentation_models_pytorch as smp


base_dir = os.path.dirname(os.path.abspath(__file__))
test_folder = os.path.join(base_dir, "data", "images", "photos_2025_RacineCnty")
output_csv = os.path.join(base_dir, "unlabeled_predictions.csv")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)


model_path = os.path.join(base_dir, "bluff_model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

results = []

if not os.path.isdir(test_folder):
    raise FileNotFoundError(f"Test folder not found: {test_folder}")

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(test_folder, img_name)

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        pred_logits = model(image_tensor)
        pred_probs = torch.sigmoid(pred_logits)
        pred_mask = (pred_probs > 0.5).float()

        pred_mask = pred_mask.squeeze().numpy()
        white_percent = pred_mask.mean() * 100

        results.append({
            "image_name": img_name,
            "predicted_white_percent": white_percent
        })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(df)
print(f"Saved results to {output_csv}")