# Dialogue Generation Example
  
## Example Card
  
- Dataset
  - Name: AI Hub SNS Dataset
  - Language: Korean
  - Data Size: 1.2M Session  
  - Download link: [[link]](https://aihub.or.kr/aidata/30718)
- Model
  - Name: KoGPT2
  - Language: Korean
  - Link: [[link]](https://huggingface.co/skt/kogpt2-base-v2)  
- Requirements
  - pytorch
  - transformers
  - wandb
  - tqdm  
  
## Get Started
  
You can train with [AI Hub SNS Dataset](https://aihub.or.kr/aidata/30718) directly.    
If you want to train with a different dataset, you only need to create a new load_dataset function and Dataset class.    
Of course, it is also necessary to change the pre-trained model name to fit the language.  
    
- Train

```
$ python3 train.py --data_dir $DATA_DIR
```
  
- Play with model
  
```
YOUR TURN: 안녕! ㅎㅎ 
BOT: 안녕!! 
YOUR TURN: 지금 뭐하고 있었어?
BOT: 나 운동하러 왔어~
...
...
```
  
```
$ python3 play.py
```
