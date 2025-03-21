from datasets import load_dataset

dataset_path = "./multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"  

#EVQA
EVQA_ds = load_dataset(dataset_path, "EVQA_data")

EVQA_passages = load_dataset(dataset_path, "EVQA_passages")

train_passages = EVQA_passages['train_passages']
val_passages = EVQA_passages['valid_passages']
test_passages = EVQA_passages['test_passages']

train_data = EVQA_ds['train']
val_data = EVQA_ds['valid']
test_data = EVQA_ds['test']

print(test_data[0])

print(test_passages[0]) 
