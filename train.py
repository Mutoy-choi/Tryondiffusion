import torch.optim as optim
from utils import *
from dataloader import VirtualTryOnDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from parallelUNet128 import ParallelUNet

# Image transformations (resize and convert to tensor)
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resizing to 256x256 for now
    transforms.ToTensor()
])

unzip_dir = 'input_data_final'

# Initialize the dataset and dataloader
ia_dir = os.path.join(unzip_dir, "Ia")
ic_dir = os.path.join(unzip_dir, "Ic")
jp_dir = os.path.join(unzip_dir, "jp")
jg_dir = os.path.join(unzip_dir, "jg")

dataset = VirtualTryOnDataset(ia_dir, ic_dir, jp_dir, jg_dir, transform=image_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Using a batch size of 8

# Display the length of the dataset and a sample batch shape
len(dataset), next(iter(dataloader))[0].shape  # Checking the shape of the first batch of Ia images

# Constants for the model
num_channels_Ia = 3  # RGB images
num_channels_zt = 3  # Placeholder, not used in the example
human_pose_dim = 136  # 17 keypoints x 2 (x, y)
garment_pose_dim = 16  # 8 keypoints x 2 (x, y)

# Initialize the model
model = ParallelUNet(human_pose_dim, garment_pose_dim, num_channels_Ia=num_channels_Ia, num_channels_zt=num_channels_zt)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 10

# Check if GPU is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize variables to store training metrics
train_loss_history = []

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for i, (ia_images, jp_keypoints, ic_images, jg_keypoints) in enumerate(dataloader):
        # Move data to device (GPU if available, else CPU)
        ia_images, ic_images = ia_images.to(device), ic_images.to(device)
        jp_keypoints, jg_keypoints = torch.tensor(jp_keypoints).float().to(device), torch.tensor(
            jg_keypoints).float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(ia_images, ia_images, jp_keypoints, jg_keypoints, ic_images)  # Placeholder for zt, not used
        loss = criterion(outputs, ia_images)  # Placeholder for ground truth, assuming ia_images for demonstration

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update loss
        epoch_loss += loss.item()

        # Print batch loss every 10 batches
        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1}] loss: {epoch_loss / 10:.4f}")
            epoch_loss = 0.0

    # Compute and store epoch loss
    epoch_loss = epoch_loss / len(dataloader)
    train_loss_history.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Final training loss
train_loss_history[-1]