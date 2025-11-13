import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Descargar dataset público de gatos (ej: Cats vs Dogs de Kaggle)
# Usaremos solo la carpeta "cat" y "not cat" para el ejemplo

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.ImageFolder(
    root='data/train', # estructura: data/train/cat/, data/train/not_cat/
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*14*14, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

torch.save(model.state_dict(), "cnn_cat.pth")
