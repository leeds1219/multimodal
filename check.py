from datasets import load_dataset, load_from_disk
from datasets import DatasetDict

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

print("========= EVQA Test Data Summary =========")
print("Number of examples:", len(test_data))
print("Number of passages:", len(test_passages))

passage_contents = test_passages["passage_content"]
passage_ids = test_passages["passage_id"]