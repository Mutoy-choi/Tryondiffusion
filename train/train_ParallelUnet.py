import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from models import LightweightParallelUNet, ParallelUNet  # Assuming you have this class defined somewhere
from dataloader import ParallelUnetDataloader_AIHub  # Uncomment if CustomDataset is in a separate file
from torch.cuda.amp import autocast, GradScaler
import time
from diffusers import DDPMScheduler  # 노이즈 스케줄러 import

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
json_file = "trainexample/exampled_json_file.json"
dataset = ParallelUnetDataloader_AIHub(json_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

# Initialize model and optimizer
IMG_CHANNEL = 3

EMB_DIM = 13

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

# Initialize both models and their optimizers
model1 = ParallelUNet(EMB_DIM, parallel_config)
model2 = ParallelUNet(EMB_DIM, parallel_config)  # Assuming you have a ParallelUNet class

# Define initial learning rates
initial_lr1 = 0.0001
initial_lr2 = 0.0001

optimizer1 = optim.AdamW(model1.parameters(), lr=initial_lr1)
optimizer2 = optim.AdamW(model2.parameters(), lr=initial_lr2)

criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Move both models to the GPU
model1.to(device)
model2.to(device)

# Initialize the gradient scaler for fp16
scaler = GradScaler()

# Define the number of steps for gradient accumulation
accumulation_steps = 12  # You can adjust this value

# Define noise level for each image
def add_noise(img, noise_level=0.1):
    noise = torch.randn_like(img) * noise_level
    noisy_img = img + noise
    return noisy_img, noise

# Function to calculate SNR
def calculate_snr(signal, noise):
    signal_power = torch.mean(signal**2)
    noise_power = torch.mean(noise**2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))  # Avoid division by zero
    return snr

# Function to adjust learning rate based on SNR
def adjust_learning_rate(optimizer, snr, base_lr=0.0001, max_lr=0.001):
    lr = base_lr + (max_lr - base_lr) * (snr / 20)  # Assuming SNR is in dB and scaling it to the learning rate range
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training loop
for epoch in range(10):  # 10 epochs
    epoch_start_time = time.time()
    print(f"Epoch {epoch + 1} start time: {epoch_start_time}")
    epoch_loss = 0.0
    num_batches = 0

    optimizer1.zero_grad(set_to_none=True)
    optimizer2.zero_grad(set_to_none=True)

    # Wrap your dataloader with tqdm for a progress bar
    for i, (combined_img, person_pose, garment_pose, ic_img, org_img) in enumerate(dataloader):

        # Move data to the GPU
        combined_img = combined_img.to(device)
        person_pose = person_pose.to(device)
        garment_pose = garment_pose.to(device)
        ic_img = ic_img.to(device)
        org_img = org_img.to(device)

        # Add noise to the combined image
        noisy_img, noise = add_noise(combined_img, noise_level=0.1)  # Adjust noise_level as needed

        # Calculate SNR
        snr = calculate_snr(combined_img, noise).mean().item()

        # Adjust learning rates based on SNR
        adjust_learning_rate(optimizer1, snr, base_lr=initial_lr1, max_lr=0.001)
        adjust_learning_rate(optimizer2, snr, base_lr=initial_lr2, max_lr=0.001)

        # Enable autocast for mixed-precision training
        with autocast():
            # Model 1's forward pass and loss calculation
            output1 = model1(noisy_img, person_pose, garment_pose, ic_img)
            loss1 = criterion(output1, org_img)  # Use original person image as the target

            # Upsample output1 from 128x128 to 256x256
            output1_upsampled = F.interpolate(output1, size=(256, 256), mode='bilinear', align_corners=True)

            # Model 2's forward pass and loss calculation
            output2 = model2(output1_upsampled, person_pose, garment_pose, ic_img)  # Using output1_upsampled as input to model2
            loss2 = criterion(output2, org_img)

            # Combine the losses
            loss = loss1 + loss2
            loss = loss / accumulation_steps  # Normalize the loss because it is accumulated

        # Backpropagation using gradient accumulation
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item() * accumulation_steps}")
            scaler.step(optimizer1)  # Performs the optimizer step for model1
            scaler.step(optimizer2)  # Performs the optimizer step for model2
            scaler.update()  # Updates the scale for next iteration
            optimizer1.zero_grad()  # Reset gradients tensors for model1
            optimizer2.zero_grad()  # Reset gradients tensors for model2

        epoch_loss += loss.item() * accumulation_steps  # Accumulate the true loss
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # After the epoch ends, calculate the elapsed time
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    print(f"Time taken for Epoch {epoch + 1}: {elapsed_time:.2f} seconds")

    # Save the models
    if epoch % 2 == 0:
        torch.save(model1.state_dict(), f"lightweight_parallel_unet_model1_epoch_{epoch + 1}.pt")
        torch.save(model2.state_dict(), f"parallel_unet_model2_epoch_{epoch + 1}.pt")
        print("Models saved.")

print("Training complete.")