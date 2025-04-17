import os
import urllib.request

def download_places365_model():
    model_url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
    model_path = "resnet18_places365.pth.tar"
    label_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
    label_path = "categories_places365.txt"

    # Download model
    if not os.path.exists(model_path):
        print("Downloading ResNet-18 Places365 model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully.")

    # Download labels
    if not os.path.exists(label_path):
        print("Downloading Places365 category labels...")
        urllib.request.urlretrieve(label_url, label_path)
        print("Label file downloaded.")
    else:
        print("Label file already exists.")

if __name__ == "__main__":
    download_places365_model()
