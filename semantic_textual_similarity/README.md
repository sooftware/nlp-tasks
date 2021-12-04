# Semantic Textual Similarity Example
  
## Example Card
  
- Dataset
  - Name: SQuAD
  - Language: English
  - Data Size: 8M  
  - Link: [[link]](https://huggingface.co/datasets/openwebtext#data-fields)
- Model
  - Name: ELECTRA
  - Paper: [[link]](https://arxiv.org/abs/2003.10555)
- Requirements
  - pytorch
  - wandb
  - tqdm  
  - sklearn
  - wget
  - transformers
  
## Get Started
  
You can train with [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) directly.     
    
- Train

```
$ python3 train.py --save $SAVE_DIR
```
