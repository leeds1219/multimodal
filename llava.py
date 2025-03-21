import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

model_path = "/home/work/GOAT3/llava-v1.6-mistral-7b-hf"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  

image_path = "/home/work/GOAT3/M2KR_Images/EVQA/inat/val/00006_Animalia_Arthropoda_Arachnida_Araneae_Araneidae_Aculepeira_ceropegia/62a4421e-a6a3-40b3-9e66-a737d45d20df.jpg"
text_input = "What is this image about?"

inputs = processor(images=image_path, text=text_input, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()} 

outputs = model.generate(**inputs, max_new_tokens=20)

decoded_output = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)
