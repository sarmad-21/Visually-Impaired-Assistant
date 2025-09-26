import torch
import torchvision.models
from torchvision import transforms
from PIL import Image

model = torchvision.models.resnet18(num_classes=365, weights=None)
checkpoint = torch.load("resnet18_places365.pth.tar", map_location=torch.device("cpu"))
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(state_dict)
model.eval()
with open("categories_places365.txt") as labels:
    classes = [line.strip().split(' ')[0][3:] for line in labels]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def scene_recognition(image_path="object.jpg"):
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image).unsqueeze(0)   # 1x3x224x224
    with torch.no_grad():
        output = model(transformed_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_classes = probabilities.topk(3)
    predicted_scenes = [(classes[i], top_probs[j].item()) for j, i in enumerate(top_classes)]
    for label, prob in predicted_scenes:
        print(f"{label} with {prob} probability")
    return predicted_scenes

