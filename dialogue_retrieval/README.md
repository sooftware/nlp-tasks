# Dialogue Retrieval Example
  
## Example Card
  
- Dataset
  - Name: Ubuntu Dialogue Corpus
  - Paper: [[link]](https://arxiv.org/abs/1506.08909)
  - Language: English
  - Download link: [[link]](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0)
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
  
You can train with [Ubuntu Dialogue Corpus](https://arxiv.org/abs/1506.08909) directly.    
If you want to train with a different dataset, you only need to create a new load_dataset function and Dataset class.    
Of course, it is also necessary to change the pre-trained model name to fit the language.  
    
- Train

```
$ python3 train.py --data_dir $DATA_DIR --num_poly_codes 64
```

- Interactive

```python
$ python3 interactive.py --checkpoint_path $CHECKPOINT_PATH --tokenizer_name $TOKENIZER_NAME --candidates_path $CANDIDATES_PATH 
```
