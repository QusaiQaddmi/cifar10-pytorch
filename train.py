import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()

        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {running_loss / len(train_loader):.4f} "
            f"Test Loss: {test_loss / len(test_loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    torch.save(model.state_dict(), "cifar10_mlp.pth")


if __name__ == "__main__":
    main()