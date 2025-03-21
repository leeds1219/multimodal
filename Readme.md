# Clone Repo
```bash
git clone https://github.com/leeds1219/multimodal.git
```

# Downloading Hugging Face Datasets (M2KR_Images & Multi-Task Multi-Modal Knowledge Retrieval Benchmark)

This guide provides step-by-step instructions to correctly download the `M2KR_Images` and `multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR` datasets from Hugging Face.

## **Downloading `M2KR_Images`**

### **1. Remove Existing Folder**
If there is a previously cloned folder, delete it first.
```bash
rm -rf M2KR_Images
```

### **2. Clone the Dataset**
Clone the dataset from Hugging Face.
```bash
git clone https://huggingface.co/datasets/BByrneLab/M2KR_Images
```


### **3. Navigate to the Folder**
```bash
cd M2KR_Images
```

### **4. Enable Git LFS**
Activate Git LFS for the current repository.
```bash
git lfs install --local
```

### **5. Download Git LFS Files**
```bash
git lfs pull
```

### **6. Verify Download**
```bash
ls -lh
```
Ensure that the files display their correct sizes.

---

## **Downloading `multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR`**

### **1. Remove Existing Folder**
If there is a previously cloned folder, delete it first.
```bash
rm -rf multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR
```

### **2. Clone the Dataset**
Clone the dataset from Hugging Face.
```bash
git clone https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR
```

### **3. Navigate to the Folder**
```bash
cd multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR
```

### **4. Enable Git LFS**
Activate Git LFS for the current repository.
```bash
git lfs install --local
```

### **5. Download Git LFS Files**
```bash
git lfs pull
```

### **6. Verify Download**
```bash
ls -lh
```
Ensure that the files display their correct sizes.

# Miniconda Installation Guide

## Download Miniconda
To download Miniconda for Linux (x86_64), use one of the following methods:

### Method 1: Using `wget`
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

or 

### Method 2: Using `curl`
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### Install Miniconda
```sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

```sh
eval "$(/home/work/miniconda3/bin/conda shell.bash hook)" 
```

# Setup

### Environment

Create virtualenv:
```
conda create -n multimodal python=3.10 -y
conda activate multimodal
```
Install Pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install faiss

```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

Test if faiss generate error
```
python -c "import faiss"
```

Install FLMR
```
pip install -e .
```

Install ColBERT engine
```
cd third_party/ColBERT
pip install -e .
```

Install other dependencies
```
pip install ujson gitpython easydict ninja datasets transformers
```

Check Dataset
```
python example.py
```

Download BLIP (Captioning model)
```
git clone https://huggingface.co/Salesforce/blip-image-captioning-base
```

Download LLM
```
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Download LMM
```
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

```
git clone https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
```