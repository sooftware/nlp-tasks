# Language Modeling Example
  
## Example Card
  
- Dataset
  - Name: OpenWebText
  - Language: English
  - Data Size: 8M  
  - Link: [[link]](https://huggingface.co/datasets/openwebtext#data-fields)
- Model
  - Name: GPT-2
  - Language: English
  - Link: [[link]](https://huggingface.co/gpt2-medium)  
- Requirements
  - pytorch
  - datasets  
  - transformers
  - wandb
  
## Get Started
  
You can train with [OpenWebText corpus](https://huggingface.co/datasets/openwebtext#data-fields) directly.
  
- Train

```
$ python3 train.py
```
  
- Play with model
  
```
ENTER PROMPT: The 45-year-old “highway shooter” who engaged in a 12-minute shootout with California Highway Patrol officers
Outputs:
The 45-year-old “highway shooter” who engaged in a 12-minute shootout with California Highway Patrol officers 
earlier this year now says Fox News host Glenn Beck has been an inspiration for his activity.
In a several thousand word expose for MediaMatters, Pacifica journalist John Hamilton interviewed 
the so-called highway shooter, Byron Williams, from prison.\n\nIn the interview, Williams details 
what he saw as an elaborate global conspiracy and tells the journalist — whom he sees as his 
“media advocate” — to look to specific broadcasts of Beck’s show for information on the conspiracy 
he describes. (MediaMatters says Beck’s show provided “information on the conspiracy theory 
that drove him over the edge: an intricate plot involving Barack Obama, philanthropist George Soros, 
a Brazilian oil company, and the BP disaster.”)\n\nThe release on Hamilton’s story explains 
that “Williams also points to other media figures — right-wing propagandist David Horowitz, 
and Internet conspiracist and repeated Fox News guest Alex Jones — as key sources of information 
to inspire his ‘revolution.'”\n\nWilliams is quoted as saying that “Beck would never say anything 
about a conspiracy, would never advocate violence. He’ll never do anything … of this nature. 
But he’ll give you every ounce of evidence that you could possibly need.”
```
  
```
$ python3 play.py --pretrain_model_name_or_path $PRETRAIN_MODEL_PATH --pretrain_tokenizer_name skt/kogpt2-base-v2
```
