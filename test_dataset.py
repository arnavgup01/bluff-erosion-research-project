from dataset import BluffDataset
import matplotlib.pyplot as plt

dataset = BluffDataset("data/images/original", "data/images/masked")

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image.permute(1,2,0))

plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(mask.squeeze(), cmap="gray")

plt.show()
