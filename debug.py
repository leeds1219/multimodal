# import os

# from datasets import load_dataset, load_from_disk
# from datasets import DatasetDict

# dataset_path = "./multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"  

# #EVQA
# EVQA_ds = load_dataset(dataset_path, "EVQA_data")

# EVQA_passages_ds = load_dataset(dataset_path, "EVQA_passages")

# print("========= Loading dataset =========")

# def add_path_prefix_in_img_path(example, prefix):
#     if example["img_path"] != None:
#         example["img_path"] = os.path.join(prefix, example["img_path"])
#     return example

# EVQA_ds = EVQA_ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": "/home/work/multimodal/M2KR_Images/EVQA"})

# use_split = "train"

# EVQA_ds = EVQA_ds[use_split]
# EVQA_passages_ds = EVQA_passages_ds[f"{use_split}_passages"]
# print("========= Data Summary =========")
# print("Number of examples:", len(EVQA_ds))
# print("Number of passages:", len(EVQA_passages_ds))

# import torch
# from PIL import Image
# import json

# from datasets import load_dataset

# example = EVQA_ds[0]

# print("===== Example Data =====")
# print("Image Path:", example["img_path"])
# print("Text:", example)

# Check image loading
# def check_image_loading(example):
#     img_path = example["img_path"]
#     if img_path and os.path.exists(img_path):
#         try:
#             image = Image.open(img_path).convert("RGB")
#             print(f"Successfully opened: {img_path}")
#         except Exception as e:
#             print(f"Failed to open {img_path}: {e}")
#     else:
#         print(f"Image not found: {img_path}")

# Test on first few examples
# for i in range(min(5, len(EVQA_ds))):
#     check_image_loading(EVQA_ds[i])

# Load BLIP model & processor
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
model_path = "./blip-image-captioning-base"  # Adjust if needed
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# # Caption generation function
# def generate_caption(img_path):
#     try:
#         image = Image.open(img_path).convert("RGB")
#         inputs = processor(images=image, return_tensors="pt").to(model.device)
#         out = model.generate(**inputs)
#         caption = processor.decode(out[0], skip_special_tokens=True)
#         return caption
#     except Exception as e:
#         print(f"Error processing {img_path}: {e}")
#         return None

# # Add captions to dataset
# def add_captions_to_example(example):
#     img_path = example["img_path"]
#     if img_path and os.path.exists(img_path):
#         example["captions"] = [generate_caption(img_path)]
#     else:
#         example["captions"] = []
#     return example

# # Apply function to dataset
# EVQA_ds = EVQA_ds.map(add_captions_to_example)

# # Save results to JSON
# output_data = [dict(example) for example in EVQA_ds]
# with open("evqa_captions.json", "w", encoding="utf-8") as f:
#     json.dump(output_data, f, ensure_ascii=False, indent=4)

# print("Caption generation complete. Saved to 'evqa_captions.json'.")