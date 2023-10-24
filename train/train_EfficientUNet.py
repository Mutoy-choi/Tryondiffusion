import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import EfficientUnetDataloader
from EfficientUNet import EfficientUNet


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = EfficientUnetDataloader(root_dir="path_to_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for i, (noisy_images, clean_images) in enumerate(dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(noisy_images)

        # Compute loss
        loss = criterion(outputs, clean_images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

print("Training complete!")
