---
library_name: transformers
license: mit
language:
- en
tags:
- retrieval
- multi-modal
- knowledge-based visual question answering
- FLMR
- PreFLMR
---

# PreFLMR model card

PreFLMR is an open-source model for multimodal knowledge retrieval. It is a transformer-based model that uses a combination of text and image inputs to retrieve relevant documents from a large corpus.

## Model Details

### Model Description

- **Model type:** FLMRModelForRetrieval
- **Language(s) (NLP):** English
- **License:** MIT License

### Paper and resources for more detail

- **Blog Post for quick overview:** https://www.jinghong-chen.net/preflmr-sota-open-sourced-multi/
- **Paper:** https://arxiv.org/abs/2402.08327
- **Gradio Demo:** https://u60544-b8d4-53eaa55d.westx.seetacloud.com:8443/
- **Repository:** https://github.com/LinWeizheDragon/FLMR 
- **Project Page:** https://preflmr.github.io/

## Uses

### Direct Use

This model can be used directly to retrieve documents from a large corpus using a combination of text and image input queries. The retrieval usage can be found in the [official implementation](https://github.com/LinWeizheDragon/FLMR).

### Downstream Use 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

This model can be used combined with language models to create a retrieval-augmented language model. The use for Knowledge-based VQA can be found in [RAVQA](https://github.com/linweizhedragon/retrieval-augmented-visual-question-answering) 

## How to Get Started with the Model

For details of training, indexing, and performing retrieval, please refer to [here](https://github.com/LinWeizheDragon/FLMR).

## Training datasets
The model is pre-trained on three types of tasks with a total of nine datasets:
1. Image to Text retrieval: WIT, KVQA, and CC3M
2. Question to Text retrieval: MSMARCO
3. Image & Question to Text retrieval: LLaVA, OVEN, OKVQA, Infoseek and E-VQA

These datasets were converted to retrieval format. For details on the dataset split and conversion process, please refer to the paper [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://arxiv.org/abs/2402.08327). We will release the proprocessed datasets soon.


## Evaluation datasets
We evaluate our models on WIT, LLaVA, OVEN, KVQA, IGLUE (subset of WIT), Infoseek, E-VQA, OKVQA and MSMARCO. 
| Model   | Vision Encoder | Text Encoder | Checkpoint Name   | No. Param. | WIT   | LLaVA  | OVEN  | KVQA  | IGLUE | Infoseek | E-VQA | OKVQA | MSMARCO |
|---------|----------------|--------------|-------------------------------------------------------------|-------|-------|--------|-------|-------|-------|----------|-------|--------|-------|
| PreFLMR | ViT-B          | Base-v2      | [LinWeizheDragon/PreFLMR_ViT-B](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-B) | 327M | 41.7  | 67.2   | 46.3  | 28.6  | 57.3  | 48.8 | 67.9 | 66.1 | 79.5 |
| PreFLMR | ViT-L          | Base-v2      | [LinWeizheDragon/PreFLMR_ViT-L](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) | 543M | 60.5  | 71.8   | 59.8  | 43.6  | 69.2  | 57.9 | 70.8 | 68.5 | 78.7 |
| PreFLMR | ViT-G          | Base-v2      | [LinWeizheDragon/PreFLMR_ViT-G](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) | 2.1B | 61.5  | 72.4   | 63.4  | 42.1  |71.5  | 59.6 | 73.1 | 68.6 | 78.6 |

For the evaluation metrics, WIT uses Recall@10, IGLUE uses Recall@1, and all the rest datasets use Recall@5.


## Citation 

**BibTeX:**
```
@article{Lin_Mei_Chen_Byrne_2024, 
        title={PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers}, 
        url={http://arxiv.org/abs/2402.08327}, 
        number={arXiv:2402.08327}, 
        publisher={arXiv}, 
        author={Lin, Weizhe and Mei, Jingbiao and Chen, Jinghong and Byrne, Bill}, 
        year={2024}}
```
