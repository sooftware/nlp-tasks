# Fill in the Blank Example
  
## Example Card
  
- Dataset
  - Name: WikiCorpus
  - Language: English
  - Data Size: 1.3M  
  - Link: [[link]](https://huggingface.co/datasets/wikicorpus)
- Model
  - Name: BERT
  - Language: English
  - Link: [[link]](https://huggingface.co/bert-base-uncased)  
- Requirements
  - pytorch
  - transformers
  - wandb
  - datasets
  
## Get Started
  
You can train with [WikiCorpus](https://huggingface.co/datasets/wikicorpus) directly.
  
- Train

```
$ python3 train.py
```
  
- Play with model
  
```
ENTER MASKED SENTENCE: I [MASK] a cat.
Outputs:
[
    {'sequence': '<s> I have a cat.</s>', 'score': 0.38627853572368622, 'token': 3944},
    {'sequence': '<s> I hate a cat.</s>', 'score': 0.11690319329500198, 'token': 7208},
    {'sequence': '<s> I love a cat.</s>', 'score': 0.058063216507434845, 'token': 5560},
]
```
  
```
$ python3 play.py
```
