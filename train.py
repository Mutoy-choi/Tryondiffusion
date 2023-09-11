import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from parallelUNet import ParallelUNet  # Assuming you have this class defined somewhere
from Customdataloader import CustomDataset  # Uncomment if CustomDataset is in a separate file
from tqdm import tqdm

torch.cuda.empty_cache()
# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
json_file = "Data/Training/winfo_train_updated.json"
dataset = CustomDataset(json_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model and optimizer
IMG_CHANNEL = 3

EMB_DIM =51

parallel_config = {
    'garment_unet': {
        'dstack': {
            'blocks': [
                {
                    'channels': 128,
                    'repeat': 3
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'channels': 512,
                    'repeat': 6
                },
                {
                    'channels': 1024,
                    'repeat': 7
                }]
        },
        'ustack': {
            'blocks': [
                {
                    'channels': 1024,
                    'repeat': 7
                },
                {
                    'channels': 512,
                    'repeat': 6
                }]
        }
    },
    'person_unet': {
        'dstack': {
            'blocks': [
                {
                    'channels': 128,
                    'repeat': 3
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 512,
                    'repeat': 6
                },
                {

                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 1024,
                    'repeat': 7
                }]
        },
        'ustack': {
            'blocks': [
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 1024,
                    'repeat': 7
                },
                {
                    'block_type': 'FiLM_ResBlk_Self_Cross',
                    'channels': 512,
                    'repeat': 6
                },
                {
                    'channels': 256,
                    'repeat': 4
                },
                {
                    'channels': 128,
                    'repeat': 3
                }]
        }
    }
}

model = ParallelUNet(EMB_DIM, parallel_config)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Move the model to the GPU
model.to(device)

# Training loop
for epoch in range(10):  # 10 epochs
    epoch_loss = 0.0
    num_batches = 0

    # Wrap your dataloader with tqdm for a progress bar
    for combined_img, person_pose, garment_pose, ic_img, org_img in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        # Move data to the GPU
        combined_img = combined_img.to(device)
        person_pose = person_pose.to(device)
        garment_pose = garment_pose.to(device)
        ic_img = ic_img.to(device)
        org_img = org_img.to(device)

        optimizer.zero_grad()

        # Model's forward pass
        output = model(combined_img, person_pose, garment_pose, ic_img)

        # Calculate loss
        loss = criterion(output, org_img)  # Use original person image as the target

        # Backpropagation and weight update
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

print("Training complete.")