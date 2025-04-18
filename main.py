import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax
import numpy as np
import gc
from typing import List, Literal, Optional, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from math import exp
from utils import EOSReachedCriteria
from base import BaseGenerator, BaseGeneratorConfig
import argparse
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import argparse

"""
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = bool(int(os.getenv("FORCE_RESET", "0")))
"""

fixed_prompt = """
<|start_header_id|>system<|end_header_id|>

Your task is to generate a question-caption pair 
using the given image description document.

Instructions:
- The document represents an image.
- Generate a caption that accurately describes the image.
- Create a question that could be reasonably asked about the image.

Format:
{
  "caption": "A short sentence describing the image.",
  "question": "A VQA-style question about the image.",
}

You are a helpful assistant.<|eot_id|>
"""

prompt_template="""
<|start_header_id|>user<|end_header_id|>

Document:
{document}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prediction_prompt_template="""
{{
    "caption": "{caption}",
    "question": "{question}".
}}
"""

fixed_prompt_="""<|start_header_id|>system<|end_header_id|>
Your task is to answer a visual question using the provided document.

Instructions:
- The document contains information about an image.
- The question is based on the image described by the document.
- Use only the document to answer the question — do not assume anything not stated.

Format:
{
  "answer": "A short and precise answer to the question."
}

You are a helpful assistant.<|eot_id|>"""

prompt_template_="""<|start_header_id|>user<|end_header_id|>

Question:
{question}

Document:
{document}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prediction_prompt_template_="""
{{
    "answer": "{answer}".
}}
"""

def compute_log_probs(prompt_batch, output_prompt_batch, fixed_prompt, tokenizer, model, llm_batch_size=16):
    combined_prompt_batch = [
        fixed_prompt + p + o for p, o in zip(prompt_batch, output_prompt_batch)
    ]
    
    split_batches = [
        combined_prompt_batch[i:i + llm_batch_size]
        for i in range(0, len(combined_prompt_batch), llm_batch_size)
    ]
    
    total_log_prob_lists = []

    for batch_idx, batch in enumerate(tqdm(split_batches, desc="Computing log probs", leave=True)):
        encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoded_batch['input_ids'],
                attention_mask=encoded_batch['attention_mask'],
                use_cache=True,
            )

        seq_len = encoded_batch['input_ids'].shape[1]
        fixed_len = len(tokenizer(fixed_prompt, padding=False, truncation=False, add_special_tokens=False)['input_ids'])

        for i in range(encoded_batch['input_ids'].shape[0]):
            input_ids = encoded_batch['input_ids'][i]
            prompt_tokens = tokenizer(prompt_batch[i], padding=False, truncation=False, add_special_tokens=False)['input_ids']
            output_prompt_tokens = tokenizer(output_prompt_batch[i], padding=False, truncation=False, add_special_tokens=False)['input_ids']
            prompt_len = len(prompt_tokens)
            output_len = len(output_prompt_tokens)

            # 실제 input_ids 계산
            real_input_ids = tokenizer(fixed_prompt + prompt_batch[i] + output_prompt_batch[i], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            pad_len = seq_len - real_input_ids.shape[0]

            # output_start_idx 계산 (padding을 고려한 시작 위치)
            output_start_idx = pad_len + fixed_len + prompt_len - 1  # causal shift 고려해서 -1

            output_logits = outputs.logits[i, output_start_idx:, :]
            output_target_ids = input_ids[output_start_idx:].contiguous()
            output_mask = encoded_batch['attention_mask'][i, output_start_idx:].contiguous()

            log_probs = torch.log(F.softmax(output_logits, dim=-1).gather(dim=-1, index=output_target_ids.unsqueeze(-1)))
            masked_log_probs = log_probs.squeeze(-1) * output_mask
            cross_entropy = -masked_log_probs
            log_prob = -cross_entropy.sum(dim=-1) / output_mask.sum(dim=-1)
            total_log_prob_lists.append(log_prob.item())

    return total_log_prob_lists

def load_data(args):
    model_path = os.path.join("/workspace", "Llama-3.1-8B-Instruct")
    retrieved_document_path = os.path.join("/workspace", "FLMR", "examples",
                                            f"{args.dataset_name}_{args.dataset_name}_{args.split}_PreFLMR_ViT-L.json")
    caption_path = os.path.join("/workspace", "captions",
                                f"{args.dataset_name}_{args.split}_with_captions.json")
    dataset_path = os.path.join("/workspace", "multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR")
    
    with open(retrieved_document_path, "r") as f:
        retrieved_docs = json.load(f)

    with open(caption_path, "r") as f:
        captions = [json.loads(line) for line in f]

    dataset = load_dataset(dataset_path, f"{args.dataset_name}_data")[args.split]
    passage_dataset = load_dataset(dataset_path, f"{args.dataset_name}_passages")
    test_passages = passage_dataset[f"{args.split}_passages"]

    passage_dict = {p["passage_id"]: p for p in test_passages}
    
    return retrieved_docs, captions, dataset, passage_dict

def initialize_model(args):
    model_path = os.path.join("/workspace", "Llama-3.1-8B-Instruct")
    hf_device_map = f"cuda:{args.cuda_device}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=hf_device_map)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.padding_side = 'left'
    
    return tokenizer, model

# def qid_to_idx(args, qid):
#     return int(qid.replace(f"{args.dataset_name}_", ""))

def static_retrieve(retrieved_docs, passage_dict, question_ids, k=5):
    """
    Retrieve documents for the provided list of question_ids from the retrieved_docs dictionary.
    This function assumes that `retrieved_docs` contains the documents and passage information.
    
    Args:
    - question_ids (list): A list of question_ids for which we want to retrieve the documents.
    - k (int): The number of passages to retrieve for each question_id (default is 5).
    
    Returns:
    - dict: A dictionary where the key is the question_id, and the value is a list of documents.
    """
    retrieved_docs_for_batch = {}

    for qid in question_ids:
        docs = retrieved_docs.get(qid, {})
        
        if docs:
            retrieved_passages = docs.get("retrieved_passage", [])[:k]  # Get top k retrieved passages
            document_list = []
            
            # For each of the retrieved passages, get the document content
            for passage in retrieved_passages:
                passage_id = passage.get('passage_id', '')
                document_ = passage_dict.get(passage_id, {})
                document_content = document_.get('passage_content', "")
                
                # Store the document content in the list
                document_list.append(document_content)

            retrieved_docs_for_batch[qid] = document_list

    return retrieved_docs_for_batch

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the EVQA model")

    # Dataset and Split
    parser.add_argument("--dataset_name", type=str, default="EVQA", help="Name of the dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train/test/validation)")

    # CUDA Device
    parser.add_argument("--cuda_device", type=int, default=7, help="CUDA device to use")

    # Retriever batch size
    parser.add_argument("--query_batch_size", type=int, default=8, help="Batch size for retriever queries")

    # LLM batch size
    parser.add_argument("--llm_batch_size", type=int, default=16, help="Batch size for LLM")

    # Token limits
    parser.add_argument("--max_total_tokens", type=int, default=4096, help="Maximum total tokens for input")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--min_new_tokens", type=int, default=1, help="Minimum number of new tokens to generate")

    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for the model")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for the model")

    # Token handling
    parser.add_argument("--truncation", action='store_true', help="Enable token truncation")
    parser.add_argument("--padding", action='store_true', help="Enable padding")

    # Stop string handling
    parser.add_argument("--include_stop_str_in_output", action='store_true', help="Include stop string in output")
    
    # GPU memory utilization
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization ratio")

    # EOS Text
    parser.add_argument("--eos_text", type=str, default=None, help="End of sequence text")

    # Model name
    parser.add_argument("--model_name", type=str, default="/workspace/Llama-3.1-8B-Instruct", help="Path to the model")

    # Retrieval
    parser.add_argument("--doc_num", type=str, default=20, help="Top k document to retrieve")

    # Caption
    parser.add_argument("--cap_num", type=str, default=5, help="Top c caption to use")

    return parser.parse_args()

def main(args):
    # Load necessary data
    retrieved_docs, captions, dataset, passage_dict = load_data(args)
    
    # Initialize model and tokenizer
    tokenizer, model = initialize_model(args)

    # Define constants and variables
    keys = list(retrieved_docs.keys())
    k = args.doc_num
    ############################################################################################################
    result_path = os.path.join("/workspace", "results", f"retrieved_docs_{args.dataset_name}_{args.split}.json")
    
    # Read existing data (if any) # this doesn't work need to fix
    existing_data = {}
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Assuming each line is a dictionary (like { ... })
                        item = eval(line)  # Convert string to dictionary
                        if isinstance(item, dict) and 'question_id' in item:
                            existing_data[item['question_id']] = item
                    except Exception as e:
                        print(f"Error parsing line: {line}, error: {e}")
                        continue
    ############################################################################################################                
    failed_indexes = []
    ############################################################################################################ 
    # Initialize the list to store question_ids
    question_ids = []
    # Iterate through batches
    for batch_start in tqdm(range(0, len(captions), args.query_batch_size), desc="Processing caption batches"):
        batch = captions[batch_start:batch_start + args.query_batch_size]
        prompt_batch = []
        output_prompt_batch = []
        prompt_batch_ = []
        output_prompt_batch_ = []

        key_indices = []
        # Collect the question_ids from the batch
        for cap in batch:
            qid = cap.get("question_id", "")
            
            # If the question_id is already in existing_data, skip the calculation and move to the next batch
            if qid in existing_data:
                continue  # Skip this item if it's already in existing_data

            question_ids.append(qid)  # Collect question_id

        # Now, retrieve documents based on question_ids
        retrieved_docs_for_batch = static_retrieve(retrieved_docs, passage_dict, question_ids, k) # Assuming retrieve function returns docs for each question_id

        for cap in batch:
            qid = cap.get("question_id", "")
            
            # Skip if it's already in existing_data
            if qid in existing_data:
                continue

            # Retrieve the documents for the current question_id
            docs = retrieved_docs_for_batch.get(qid, [])

            # Ensure docs is a list, if not, continue
            if not docs:
                continue
            
            # Get the index for this question_id
            qid_to_index = { 
                str(example["question_id"]): idx
                for idx, example in enumerate(dataset)
            }
            i = qid_to_index.get(qid)
            
            key_indices.append(i)
            
            # Retrieve question and answer for the current index
            question = dataset[i]["question"]
            answer = dataset[i]["gold_answer"]

            # Now process the retrieved documents
            for idx in range(k):
                # Check if the idx is valid for docs list
                if idx < len(docs):
                    document_content = docs[idx]

                    # Prepare the prompts
                    prompt = prompt_template.format(document=document_content)
                    prompt_ = prompt_template_.format(question=question, document=document_content)
                    output_prompt_ = prediction_prompt_template_.format(answer=answer)

                    prompt_batch_.append(prompt)
                    output_prompt_batch_.append(output_prompt_)

                    # Loop through captions and generate prompts for each
                    caption_list = cap["captions"][:args.cap_num]
                    for caption in caption_list:
                        output_prompt = prediction_prompt_template.format(caption=caption, question=question)
                        prompt_batch.append(prompt)
                        output_prompt_batch.append(output_prompt)
        try:
            # Try computing log probabilities
            log_probs = compute_log_probs(prompt_batch, output_prompt_batch, fixed_prompt, tokenizer, model, args.llm_batch_size)
            log_probs_ = compute_log_probs(prompt_batch_, output_prompt_batch_, fixed_prompt_, tokenizer, model, args.llm_batch_size)
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU Out of Memory error occurred at batch {batch_start}! Skipping this batch and continuing.")
            torch.cuda.empty_cache()  # Clear the GPU memory cache
            failed_indexes.append([batch_start + cap_idx for cap_idx in range(len(batch))])
            continue  # Skip the current batch and move to the next batch
        
        # log_probs = compute_log_probs(prompt_batch, output_prompt_batch, fixed_prompt, tokenizer, model, args.llm_batch_size)
        # log_probs_ = compute_log_probs(prompt_batch_, output_prompt_batch_, fixed_prompt_, tokenizer, model, args.llm_batch_size)
        ###########################################################################################################################

        # Save log probabilities
        for b_idx, i in enumerate(key_indices):
            for idx in range(k):
                target_base = b_idx * k * args.cap_num  
                gold_idx = b_idx * k  

                target_idx = target_base + idx * args.cap_num
                avg_log_prob = sum(log_probs[target_idx:target_idx + args.cap_num]) / args.cap_num
                full_log_prob = avg_log_prob + log_probs_[gold_idx + idx]

                retrieved_docs[keys[i]]["retrieved_passage"][:k][idx]["log_prob"] = full_log_prob
        
        with open(result_path, "a") as f:  # "a" mode for append
            for batch_item in batch:
                qid = batch_item.get("question_id", "")
                if qid not in existing_data:  # Check if the question_id already exists
                    # Save the entry with qid as the key
                    json.dump({qid: retrieved_docs[keys[i]]}, f)  # Store qid: retrieved_docs[keys[i]]
                    f.write("\n")  # Add a newline between entries
                    existing_data[qid] = retrieved_docs[keys[i]]  # Add to existing_data to avoid saving again

        print(f"Saved to {result_path}")

    # 실패한 배치들 기록
    if failed_indexes:
        print(f"Failed indexes (OOM errors) occurred at the following positions: {failed_indexes}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

    # python main.py --