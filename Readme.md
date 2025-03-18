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
