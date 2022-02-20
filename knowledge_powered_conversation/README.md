# Knowledge Powered Conversation Example
  
## Example Card
  
- Dataset
  - Name: Wizard of Wikipedia
  - Paper: [[link]](https://arxiv.org/abs/1811.01241)
  - Language: English
- Model
  - Name: Bart
  - Paper: [[link]](https://arxiv.org/abs/1910.13461)  
- Requirements
  - pytorch
  - transformers
  - wandb
  - tqdm
  
## Get Started
  
You can train with [Wizard of Wikipedia](https://arxiv.org/abs/1811.01241) directly.    
If you want to train with a different dataset, you only need to create a new load_dataset function and Dataset class.    
Of course, it is also necessary to change the pre-trained model name to fit the language.  
    
- Train

```
$ python3 train.py --pretrain_model_name_or_path facebook/bart-base --num_epochs 5
```

