# Named Entity Recognition Example
  
## Example Card
  
- Dataset
  - Name: Naver NER
  - Language: Korean
  - Data Size: 90,000
  - Dataset link: [[link]](https://github.com/naver/nlp-challenge)
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
  
You can train with [Naver NER Dataset](https://aihub.or.kr/aidata/105) directly.     
    
- Train

```
$ python3 train.py --save $SAVE_DIR
```
