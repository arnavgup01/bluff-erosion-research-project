import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import BluffDataset

dataset = BluffDataset("data/images/original", "data/images/masked")

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = smp.Unet(
    encoder_name="resnet34",      
    encoder_weights="imagenet",   
    in_channels=3,                
    classes=1                    
)

loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 5

for epoch in range(epochs):
    print("Epoch:", epoch)
    
    for images, masks in dataloader:
        
        preds = model(images)          
    
        loss = loss_fn(preds, masks)   
        
        optimizer.zero_grad()        
        loss.backward()                
        optimizer.step()               
        
    print("Loss:", loss.item())
    


model.eval()


image, true_mask = dataset[6]


image = image.unsqueeze(0)

with torch.no_grad():
    pred_logits = model(image)            
    pred_probs = torch.sigmoid(pred_logits) 
    pred_mask = (pred_probs > 0.5).float()   


image = image.squeeze(0)
true_mask = true_mask.squeeze(0)
pred_mask = pred_mask.squeeze(0).squeeze(0)


veg_percent = pred_mask.mean().item() * 100
print(f"Predicted Vegetation: {veg_percent:.2f}%")

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image.permute(1,2,0))

plt.subplot(1,3,2)
plt.title("Ground Truth Mask")
plt.imshow(true_mask, cmap="gray")

plt.subplot(1,3,3)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")

plt.show()


model.eval()

results = []

with torch.no_grad():
    for idx in range(len(dataset)):
        image, true_mask = dataset[idx]

        image_batch = image.unsqueeze(0)

        pred_logits = model(image_batch)
        pred_probs = torch.sigmoid(pred_logits)
        pred_mask = (pred_probs > 0.5).float()

        pred_mask = pred_mask.squeeze(0).squeeze(0)

        veg_percent = pred_mask.mean().item() * 100

        img_name = dataset.images[idx]

        base_name = os.path.splitext(img_name)[0]
        matching_masks = [f for f in os.listdir(dataset.mask_dir) if f.startswith(base_name + "_masked")]
        mask_name = matching_masks[0]

        if mask_name.endswith("_v.png"):
            label = "v"
        elif mask_name.endswith("_pv.png"):
            label = "pv"
        elif mask_name.endswith("_nv.png"):
            label = "nv"
        else:
            label = "unknown"

        results.append({
            "image_name": img_name,
            "mask_name": mask_name,
            "predicted_veg_percent": veg_percent,
            "label": label
        })

df = pd.DataFrame(results)


torch.save(model.state_dict(), "bluff_model.pth")

print(df)

df.to_csv("vegetation_predictions.csv", index=False)

