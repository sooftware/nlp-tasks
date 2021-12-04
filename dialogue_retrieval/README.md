# Dialogue Retrieval Example
  
## Example Card
  
- Dataset
  - Name: AI Hub SNS Dataset
  - Language: Korean
  - Data Size: 1.2M Session  
  - Download link: [[link]](https://aihub.or.kr/aidata/30718)
- Model
  - Name: Poly Encoder
  - Paper: [[link]](https://arxiv.org/abs/1905.01969)  
- Requirements
  - pytorch
  - transformers
  - wandb
  - tqdm  
  - sklearn
  
## Get Started
  
You can train with [AI Hub SNS Dataset](https://aihub.or.kr/aidata/30718) directly.    
If you want to train with a different dataset, you only need to create a new load_dataset function and Dataset class.    
Of course, it is also necessary to change the pre-trained model name to fit the language.  
    
- Train

```
$ python3 train.py --data_dir $DATA_DIR --num_poly_codes 64
```
