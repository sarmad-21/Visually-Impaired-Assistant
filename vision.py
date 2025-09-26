from camera import capture_image
from ultralytics import YOLO
from collections import Counter
from classify_scene import scene_recognition

yolo_model = YOLO("yolov8n.pt").to("cpu")

def object_detection(image_path="object.jpg"):
    result = yolo_model(image_path, verbose=False) #run object detection on image
    names = result[0].names #output for single image
    detected = result[0].boxes.cls.tolist()
    if not detected:
        return {}
    labels = [names[int(i)] for i in detected]
    label_counts = Counter(labels)
    return dict(label_counts)

def analyze_environment(image_path="object.jpg"):
    capture_image(image_path)
    objects = object_detection(image_path) # run object detection
    scenes = scene_recognition(image_path)  #run scene recogntion returns label and probability
    if scenes[0][1] > 0.6:
        scene_labels = [scenes[0][0]]
    else:
        scene_labels = [s[0] for s in scenes[:3]]
    return {
        "objects": objects,
        "scenes": scene_labels
    }

def create_llava_prompt(environment_info):
    objects_text = ", ".join(
        f"{count} {obj}" for obj, count in environment_info["objects"].items()
    )
    scenes_text = " or ".join(environment_info["scenes"])
    return (
        f"Describe this {scenes_text} scene with {objects_text}. "
        "Mention spatial relationships and safety concerns for visually impaired and blind people. "
        "Keep response under 25 words."
    )





