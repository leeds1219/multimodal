---
language:
- en
license: mit
size_categories:
- 10M<n<100M
task_categories:
- knowledge-based-visual-question-answering
- Knowledge-retrieval
- passage-retrieval
pretty_name: M2KR
dataset_info:
- config_name: CC_data
  features:
  - name: original_data_id
    sequence: string
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: 'null'
  - name: instruction
    dtype: string
  splits:
  - name: train
    num_bytes: 160122542
    num_examples: 595375
  download_size: 60703737
  dataset_size: 160122542
- config_name: CC_passages
  features:
  - name: language
    dtype: string
  - name: original_data_id
    dtype: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: train_passages
    num_bytes: 115902148
    num_examples: 595375
  download_size: 48443038
  dataset_size: 115902148
- config_name: EVQA_data
  features:
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: answers
    sequence: string
  - name: gold_answer
    dtype: string
  - name: question_type
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: train
    num_bytes: 233843951
    num_examples: 167369
  - name: valid
    num_bytes: 12191971
    num_examples: 9852
  - name: test
    num_bytes: 4958556
    num_examples: 3750
  download_size: 39851691
  dataset_size: 250994478
- config_name: EVQA_passages
  features:
  - name: language
    dtype: string
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: train_passages
    num_bytes: 58570897
    num_examples: 50205
  - name: valid_passages
    num_bytes: 59117345
    num_examples: 50753
  - name: test_passages
    num_bytes: 60113716
    num_examples: 51472
  download_size: 106160568
  dataset_size: 177801958
- config_name: IGLUE_data
  features:
  - name: question_id
    dtype: string
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: instruction
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: test
    num_bytes: 1188601
    num_examples: 685
  download_size: 634409
  dataset_size: 1188601
- config_name: IGLUE_passages
  features:
  - name: language
    dtype: string
  - name: page_url
    dtype: string
  - name: image_url
    dtype: string
  - name: page_title
    dtype: string
  - name: section_title
    dtype: string
  - name: hierarchical_section_title
    dtype: string
  - name: caption_reference_description
    dtype: string
  - name: caption_attribution_description
    dtype: string
  - name: caption_alt_text_description
    dtype: string
  - name: mime_type
    dtype: string
  - name: original_height
    dtype: int64
  - name: original_width
    dtype: int64
  - name: is_main_image
    dtype: bool
  - name: attribution_passes_lang_id
    dtype: bool
  - name: page_changed_recently
    dtype: bool
  - name: context_page_description
    dtype: string
  - name: context_section_description
    dtype: string
  - name: image_id
    dtype: string
  - name: original_data_id
    dtype: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_downloaded
    dtype: bool
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: test_passages
    num_bytes: 3595283
    num_examples: 1000
  download_size: 2072916
  dataset_size: 3595283
- config_name: Infoseek_data
  features:
  - name: question_id
    dtype: string
  - name: image_id
    dtype: string
  - name: question
    dtype: string
  - name: answers
    sequence: string
  - name: answer_eval
    sequence: string
  - name: data_split
    dtype: string
  - name: wikidata_value
    dtype: float64
  - name: wikidata_range
    sequence: float64
  - name: entity_id
    dtype: string
  - name: entity_text
    dtype: string
  - name: image_path
    dtype: string
  - name: gold_answer
    dtype: string
  - name: objects
    list:
    - name: attribute_scores
      sequence: float64
    - name: attributes
      sequence: string
    - name: class
      dtype: string
    - name: ocr
      sequence: 'null'
    - name: rect
      sequence: float64
  - name: related_item_ids
    sequence: string
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: ROIs
    sequence: 'null'
  - name: found
    dtype: bool
  - name: img_caption
    dtype: string
  - name: instruction
    dtype: string
  - name: img_path
    dtype: string
  - name: question_type
    dtype: string
  splits:
  - name: train
    num_bytes: 10097646987
    num_examples: 676441
  - name: test
    num_bytes: 77721658
    num_examples: 4708
  download_size: 3494936536
  dataset_size: 10175368645
- config_name: Infoseek_passages
  features:
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: train_passages
    num_bytes: 67381873
    num_examples: 98276
  - name: test_passages
    num_bytes: 67381873
    num_examples: 98276
  download_size: 79086526
  dataset_size: 134763746
- config_name: KVQA_data
  features:
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: question_id
    dtype: string
  - name: instruction
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 36180062
    num_examples: 64396
  - name: valid
    num_bytes: 7651029
    num_examples: 13365
  - name: test
    num_bytes: 2969856
    num_examples: 5120
  download_size: 5307195
  dataset_size: 46800947
- config_name: KVQA_passages
  features:
  - name: language
    dtype: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: valid_passages
    num_bytes: 2148876
    num_examples: 4648
  - name: train_passages
    num_bytes: 7287243
    num_examples: 16215
  - name: test_passages
    num_bytes: 2148876
    num_examples: 4648
  download_size: 4755781
  dataset_size: 11584995
- config_name: LLaVA_data
  features:
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: llava_split
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: train
    num_bytes: 259696568
    num_examples: 350747
  - name: test
    num_bytes: 4429239
    num_examples: 5120
  download_size: 110447927
  dataset_size: 264125807
- config_name: LLaVA_passages
  features:
  - name: language
    dtype: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  - name: llava_split
    dtype: string
  splits:
  - name: train_passages
    num_bytes: 201390688
    num_examples: 350747
  - name: test_passages
    num_bytes: 4259479
    num_examples: 6006
  download_size: 95290912
  dataset_size: 205650167
- config_name: MSMARCO_data
  features:
  - name: original_data_id
    sequence: string
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: 'null'
  - name: img_path
    dtype: 'null'
  - name: image_id
    dtype: 'null'
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: train
    num_bytes: 211125342
    num_examples: 400782
  - name: valid
    num_bytes: 3558848
    num_examples: 6980
  - name: test
    num_bytes: 2623416
    num_examples: 5120
  download_size: 120209939
  dataset_size: 217307606
- config_name: MSMARCO_passages
  features:
  - name: language
    dtype: string
  - name: original_data_id
    dtype: string
  - name: img_id
    dtype: 'null'
  - name: img_path
    dtype: 'null'
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: valid_passages
    num_bytes: 151114792
    num_examples: 400000
  - name: train_passages
    num_bytes: 3343395078
    num_examples: 8841823
  - name: test_passages
    num_bytes: 151114792
    num_examples: 400000
  download_size: 1954619356
  dataset_size: 3645624662
- config_name: OKVQA_data
  features:
  - name: answers
    sequence: string
  - name: gold_answer
    dtype: string
  - name: question
    dtype: string
  - name: question_id
    dtype: string
  - name: img_path
    dtype: string
  - name: img_key_full
    dtype: string
  - name: img_key
    dtype: int64
  - name: img_file_name
    dtype: string
  - name: img
    dtype: 'null'
  - name: img_caption
    struct:
    - name: caption
      dtype: string
    - name: conf
      dtype: float64
  - name: objects
    list:
    - name: attribute_scores
      sequence: float64
    - name: attributes
      sequence: string
    - name: class
      dtype: string
    - name: ocr
      list:
      - name: score
        dtype: float64
      - name: text
        dtype: string
    - name: rect
      sequence: float64
  - name: img_ocr
    list:
    - name: description
      dtype: string
    - name: vertices
      sequence:
        sequence: int64
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: related_item_ids
    sequence: string
  - name: __index_level_0__
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: train
    num_bytes: 174828614
    num_examples: 9009
  - name: valid
    num_bytes: 97313755
    num_examples: 5046
  - name: test
    num_bytes: 97313678
    num_examples: 5046
  download_size: 107113939
  dataset_size: 369456047
- config_name: OKVQA_passages
  features:
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: valid_passages
    num_bytes: 78929116
    num_examples: 114809
  - name: train_passages
    num_bytes: 78929116
    num_examples: 114809
  - name: test_passages
    num_bytes: 78929116
    num_examples: 114809
  download_size: 136470207
  dataset_size: 236787348
- config_name: OVEN_data
  features:
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: wiki_entity
    dtype: string
  - name: wiki_entity_id
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: train
    num_bytes: 380210407
    num_examples: 339137
  - name: valid
    num_bytes: 133191336
    num_examples: 119136
  - name: test
    num_bytes: 5696371
    num_examples: 5120
  download_size: 83074582
  dataset_size: 519098114
- config_name: OVEN_passages
  features:
  - name: language
    dtype: string
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: valid_passages
    num_bytes: 2647627
    num_examples: 3192
  - name: train_passages
    num_bytes: 6725171
    num_examples: 7943
  - name: test_passages
    num_bytes: 2647627
    num_examples: 3192
  download_size: 7283816
  dataset_size: 12020425
- config_name: WIT_data
  features:
  - name: original_data_id
    sequence: string
  - name: pos_item_ids
    sequence: string
  - name: pos_item_contents
    sequence: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_id
    dtype: string
  - name: question_id
    dtype: string
  - name: instruction
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 4689765006
    num_examples: 2810679
  - name: valid
    num_bytes: 35765246
    num_examples: 19994
  - name: test
    num_bytes: 8890482
    num_examples: 5120
  download_size: 2498894567
  dataset_size: 4734420734
- config_name: WIT_passages
  features:
  - name: language
    dtype: string
  - name: page_url
    dtype: string
  - name: image_url
    dtype: string
  - name: page_title
    dtype: string
  - name: section_title
    dtype: string
  - name: hierarchical_section_title
    dtype: string
  - name: caption_reference_description
    dtype: string
  - name: caption_attribution_description
    dtype: string
  - name: caption_alt_text_description
    dtype: string
  - name: mime_type
    dtype: string
  - name: original_height
    dtype: int64
  - name: original_width
    dtype: int64
  - name: is_main_image
    dtype: bool
  - name: attribution_passes_lang_id
    dtype: bool
  - name: page_changed_recently
    dtype: bool
  - name: context_page_description
    dtype: string
  - name: context_section_description
    dtype: string
  - name: image_id
    dtype: string
  - name: original_data_id
    dtype: string
  - name: img_id
    dtype: string
  - name: img_path
    dtype: string
  - name: image_downloaded
    dtype: bool
  - name: passage_id
    dtype: string
  - name: passage_content
    dtype: string
  splits:
  - name: valid_passages
    num_bytes: 132381872
    num_examples: 39478
  - name: train_passages
    num_bytes: 13419201634
    num_examples: 4120010
  - name: test_passages
    num_bytes: 132381872
    num_examples: 39478
  download_size: 8424698596
  dataset_size: 13683965378
configs:
- config_name: CC_data
  data_files:
  - split: train
    path: CC_data/train-*
- config_name: CC_passages
  data_files:
  - split: train_passages
    path: CC_passages/train_passages-*
- config_name: EVQA_data
  data_files:
  - split: train
    path: EVQA_data/train-*
  - split: valid
    path: EVQA_data/valid-*
  - split: test
    path: EVQA_data/test-*
- config_name: EVQA_passages
  data_files:
  - split: train_passages
    path: EVQA_passages/train_passages-*
  - split: valid_passages
    path: EVQA_passages/valid_passages-*
  - split: test_passages
    path: EVQA_passages/test_passages-*
- config_name: IGLUE_data
  data_files:
  - split: test
    path: IGLUE_data/test-*
- config_name: IGLUE_passages
  data_files:
  - split: test_passages
    path: IGLUE_passages/test_passages-*
- config_name: Infoseek_data
  data_files:
  - split: train
    path: Infoseek_data/train-*
  - split: test
    path: Infoseek_data/test-*
- config_name: Infoseek_passages
  data_files:
  - split: train_passages
    path: Infoseek_passages/train_passages-*
  - split: test_passages
    path: Infoseek_passages/test_passages-*
- config_name: KVQA_data
  data_files:
  - split: train
    path: KVQA_data/train-*
  - split: valid
    path: KVQA_data/valid-*
  - split: test
    path: KVQA_data/test-*
- config_name: KVQA_passages
  data_files:
  - split: valid_passages
    path: KVQA_passages/valid_passages-*
  - split: train_passages
    path: KVQA_passages/train_passages-*
  - split: test_passages
    path: KVQA_passages/test_passages-*
- config_name: LLaVA_data
  data_files:
  - split: train
    path: LLaVA_data/train-*
  - split: test
    path: LLaVA_data/test-*
- config_name: LLaVA_passages
  data_files:
  - split: train_passages
    path: LLaVA_passages/train_passages-*
  - split: test_passages
    path: LLaVA_passages/test_passages-*
- config_name: MSMARCO_data
  data_files:
  - split: train
    path: MSMARCO_data/train-*
  - split: valid
    path: MSMARCO_data/valid-*
  - split: test
    path: MSMARCO_data/test-*
- config_name: MSMARCO_passages
  data_files:
  - split: valid_passages
    path: MSMARCO_passages/valid_passages-*
  - split: train_passages
    path: MSMARCO_passages/train_passages-*
  - split: test_passages
    path: MSMARCO_passages/test_passages-*
- config_name: OKVQA_data
  data_files:
  - split: train
    path: OKVQA_data/train-*
  - split: valid
    path: OKVQA_data/valid-*
  - split: test
    path: OKVQA_data/test-*
- config_name: OKVQA_passages
  data_files:
  - split: valid_passages
    path: OKVQA_passages/valid_passages-*
  - split: train_passages
    path: OKVQA_passages/train_passages-*
  - split: test_passages
    path: OKVQA_passages/test_passages-*
- config_name: OVEN_data
  data_files:
  - split: train
    path: OVEN_data/train-*
  - split: valid
    path: OVEN_data/valid-*
  - split: test
    path: OVEN_data/test-*
- config_name: OVEN_passages
  data_files:
  - split: valid_passages
    path: OVEN_passages/valid_passages-*
  - split: train_passages
    path: OVEN_passages/train_passages-*
  - split: test_passages
    path: OVEN_passages/test_passages-*
- config_name: WIT_data
  data_files:
  - split: train
    path: WIT_data/train-*
  - split: valid
    path: WIT_data/valid-*
  - split: test
    path: WIT_data/test-*
- config_name: WIT_passages
  data_files:
  - split: valid_passages
    path: WIT_passages/valid_passages-*
  - split: train_passages
    path: WIT_passages/train_passages-*
  - split: test_passages
    path: WIT_passages/test_passages-*
---



# PreFLMR M2KR Dataset Card

## Dataset details

**Dataset type:**
M2KR is a benchmark dataset for multimodal knowledge retrieval. It contains a collection of tasks and datasets for training and evaluating multimodal knowledge retrieval models.

We pre-process the datasets into a uniform format and write several task-specific prompting instructions for each dataset. The details of the instruction can be found in the paper. The M2KR benchmark contains three types of tasks:
#### Image to Text (I2T) retrieval
These tasks evaluate the ability of a retriever to find relevant documents associated with an input image.   
Component tasks are WIT, IGLUE-en, KVQA, and CC3M.  

#### Question to Text (Q2T) retrieval
This task is based on MSMARCO and is included to assess whether multi-modal retrievers retain their ability in text-only retrieval after any retraining for images. 

#### Image & Question to Text (IQ2T) retrieval
This is the most challenging task which requires joint understanding of questions and images for accurate retrieval.  It consists of these subtasks:  
OVEN, LLaVA, OKVQA, Infoseek and E-VQA.


**Paper or resources for more information:**
- **Paper:** https://arxiv.org/abs/2402.08327
- **Project Page:** https://preflmr.github.io/
- **Huggingface Implementation:** https://github.com/LinWeizheDragon/FLMR
For details on the example usage of the dataset, please see the [M2KR Benchmark Datasets](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md)

We release the raw images used in M2KR benchmark, please see the [M2kR Benchmark Images](https://huggingface.co/datasets/BByrneLab/M2KR_Images)

**License:**
MIT License

**Where to send questions or comments about the model:**
https://github.com/LinWeizheDragon/FLMR/issues

## Intended use
**Primary intended uses:**
The primary use of M2KR is for pretraining general-purpose multimodal knowledge retrieval models and benchmarking their performance.

**Primary intended users:**
The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

**Citation**
If our work helped your research, please kindly cite our paper for PreFLMR.
```
       
@inproceedings{lin-etal-2024-preflmr,
    title = "{P}re{FLMR}: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers",
    author = "Lin, Weizhe  and
      Mei, Jingbiao  and
      Chen, Jinghong  and
      Byrne, Bill",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.289",
    pages = "5294--5316",
    abstract = "Large Multimodal Models (LMMs) excel in natural language and visual understanding but are challenged by exacting tasks such as Knowledge-based Visual Question Answering (KB-VQA) which involve the retrieval of relevant information from document collections to use in shaping answers to questions. We present an extensive training and evaluation framework, M2KR, for KB-VQA. M2KR contains a collection of vision and language tasks which we have incorporated into a single suite of benchmark tasks for training and evaluating general-purpose multi-modal retrievers. We use M2KR to develop PreFLMR, a pre-trained version of the recently developed Fine-grained Late-interaction Multi-modal Retriever (FLMR) approach to KB-VQA, and we report new state-of-the-art results across a range of tasks. We also present investigations into the scaling behaviors of PreFLMR intended to be useful in future developments in general-purpose multi-modal retrievers.",
}

```
