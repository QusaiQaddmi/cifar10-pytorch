import sys
import torch
from PIL import Image
from torchvision import transforms
from train import MLP


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP().to(device)
    model.load_state_dict(
        torch.load(
            "cifar10_mlp.pth",
            map_location=device
        )
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    print(CLASSES[prediction])


if __name__ == "__main__":
    predict(sys.argv[1])