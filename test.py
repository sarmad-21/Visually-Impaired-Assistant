from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
from vision import analyze_environment, create_llava_prompt

env_info = analyze_environment("object.jpg")
prompt = create_llava_prompt(env_info)

image = Image.open("object.jpg").convert("RGB")
model_id = "llava-hf/llava-phi"  # ✅ working public model

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id)

inputs = processor(prompt, image, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=50)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n📝 LLaVA Description:")
print(output)
