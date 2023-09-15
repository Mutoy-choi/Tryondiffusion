import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from lightparallelUNet import LightweightParallelUNet # Assuming you have this class defined somewhere
from parallelUNet import ParallelUNet
from Customdataloader import CustomDataset  # Uncomment if CustomDataset is in a separate file
from torch.cuda.amp import autocast, GradScaler
import time

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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)


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

# Initialize both models and their optimizers
model1 = LightweightParallelUNet(EMB_DIM, parallel_config)
model2 = ParallelUNet(EMB_DIM, parallel_config)  # Assuming you have a ParallelUNet class

optimizer1 = optim.AdamW(model1.parameters(), lr=0.0001)
optimizer2 = optim.AdamW(model2.parameters(), lr=0.0001)

criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Move both models to the GPU
model1.to(device)
model2.to(device)


# Initialize the gradient scaler for fp16
scaler = GradScaler()

# Define the number of steps for gradient accumulation
accumulation_steps = 12  # You can adjust this value

model.load_state_dict(torch.load('lightweight_parallel_unet_model_epoch_1.pt'))
# Training loop
for epoch in range(10):  # 10 epochs
    epoch_start_time = time.time()
    print(epoch_start_time)
    epoch_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    # Wrap your dataloader with tqdm for a progress bar
    for i, (combined_img, person_pose, garment_pose, ic_img, org_img) in enumerate(dataloader):

        # Move data to the GPU and convert to fp16
        combined_img = combined_img.to(device)
        person_pose = person_pose.to(device)
        garment_pose = garment_pose.to(device)
        ic_img = ic_img.to(device)
        org_img = org_img.to(device)

        # Enable autocast for mixed-precision training
        with autocast():
            # Model 1's forward pass and loss calculation
            output1 = model1(combined_img, person_pose, garment_pose, ic_img)
            loss1 = criterion(output1, org_img)  # Use original person image as the target
    
            # Model 2's forward pass and loss calculation
            output2 = model2(output1, person_pose, garment_pose, ic_img)  # Using output1 as input to model2
            loss2 = criterion(output2, org_img)

            # Combine the losses if needed
            loss = loss1 + loss2
            loss = loss / accumulation_steps  # Normalize the loss because it is accumulated

        # Backpropagation using gradient accumulation
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item() * accumulation_steps}")
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            scaler.step(optimizer)  # Performs the optimizer step
            scaler.update()  # Updates the scale for next iteration
            optimizer.zero_grad()  # Reset gradients tensors

        epoch_loss += loss.item() * accumulation_steps  # Accumulate the true loss
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # After the epoch ends, calculate the elapsed time
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    print(f"Time taken for Epoch {epoch + 1}: {elapsed_time:.2f} seconds")
    if epoch%2 == 0:
        # Save the model after training is complete
        torch.save(model.state_dict(), f"lightweight_parallel_unet_model_2_epoch_{epoch+1}.pt")

        print("model saved.")

print("Training complete.")

# Save the model after training is complete
torch.save(model.state_dict(), "lightweight_parallel_unet_model.pt")

print("Training complete and model saved.")


