# Speech Recognition Example
  
## Example Card
  
- Dataset
  - Name: KsponSpeech
  - Language: Korean
  - Data Size: 1,000 hours
  - Download link: [[link]](https://aihub.or.kr/aidata/105)
- Model
  - Name: Listen, Attend and Spell
  - Paper: [[link]](https://arxiv.org/abs/1508.01211)
- Requirements
  - pytorch
  - wandb
  - tqdm  
  - python-Levenshtein  
  
## Get Started
  
You can train with [KsponSpeech](https://aihub.or.kr/aidata/105) directly.     
    
- Train

```
$ python3 train.py \ 
    --data_dir $DATA_DIR \
    --test_data_dir $TEST_DATA_DIR \
    --test_manifest_path $TEST_MANIFEST_PATH
```
  
- Play with model
  
```
AUDIO_PATH: example1.pcm 
Outputs:
이게 정말 1 퍼센트 가정의 모습이야?
```
  
```
$ python3 play.py
```
