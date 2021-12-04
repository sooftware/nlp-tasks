# Natural Language Processing Tasks and Examples
   
With the advancement of A.I. technology in recent years, natural language processing technology has been able to solve so many problems. While working as an NLP engineer, I encountered various tasks, and I thought it would be nice to gather and organize the natural language processing tasks I have dealt with in one place. Borrowing [Kyubyong's project](https://github.com/Kyubyong/nlp-tasks) format, I organized natural language processing tasks with references and example code.
  
## Automated Essay Scoring
  
- **`WIKI`** [Automated Essay Scoring](https://en.wikipedia.org/wiki/Automated_essay_scoring)  
- **`DATA`** [The Hewlett Foundation: Automated Essay Scoring](https://www.kaggle.com/c/asap-aes/data)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`MODEL`** [Electra](https://arxiv.org/abs/2003.10555)
- **`OFF-THE-SHELF`** [Pororo's AES](https://kakaobrain.github.io/pororo/text_cls/aes.html)
  
## Automatic Speech Recognition
  
- **`WIKI`** [Speech Recognition](https://en.wikipedia.org/wiki/Speech_recognition)
- **`DATA`** [LibriSpeech](https://www.openslr.org/12)
- **`DATA`** [AISHELL-1](https://arxiv.org/abs/1709.05522)
- **`DATA`** [KsponSpeech](https://aihub.or.kr/aidata/105)
- **`MODEL`** [Deep Speech2](https://arxiv.org/abs/1512.02595)
- **`MODEL`** [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
- **`MODEL`** [Wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- **`OFF-THE-SHELF`** [Pororo's ASR](https://kakaobrain.github.io/pororo/miscs/asr.html)
- **`CODE`** [Example with KsponSpeech](https://github.com/sooftware/nlp-tasks/tree/main/automatic_speech_recognition)
  
## Dialogue Generation

- **`WIKI`** [Dialogue System](https://en.wikipedia.org/wiki/Dialogue_system)
- **`DATA`** [Persona Chat](https://github.com/facebookresearch/ParlAI/tree/main/projects/personachat)
- **`DATA`** [Korean SNS Corpus](https://aihub.or.kr/aidata/30718)
- **`MODEL`** [Dialogue GPT](https://arxiv.org/abs/1911.00536)
- **`CODE`** [Example with Korean SNS Corpus](https://github.com/sooftware/nlp-tasks/tree/main/dialogue_generation)

## Dialogue Retrieval

- **`WIKI`** [Dialogue System](https://en.wikipedia.org/wiki/Dialogue_system)
- **`DATA`** [Persona Chat](https://github.com/facebookresearch/ParlAI/tree/main/projects/personachat)
- **`DATA`** [Korean SNS Corpus](https://aihub.or.kr/aidata/30718)
- **`MODEL`** [Poly Encoder](https://arxiv.org/abs/1905.01969)
- **`CODE`** [Example with Korean SNS Corpus](https://github.com/sooftware/nlp-tasks/tree/main/dialogue_retrieval)

## Fill in the Blank
  
- **`WIKI`** [Cloze Test](https://en.wikipedia.org/wiki/Cloze_test)
- **`INFO`** [Masked-Language-Modeling with BERT](https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`OFF-THE-SHELF`** [Pororo's Fill in the Blank](https://kakaobrain.github.io/pororo/tagging/fill.html)
- **`CODE`** [Example with WikiCorpus](https://github.com/sooftware/nlp-tasks/tree/main/fill_in_the_blank)
  
## Grammatical Error Correction
  
- **`WIKI`** [Autocorrection](https://en.wikipedia.org/wiki/Autocorrection)
- **`DATA`** [NUS Non-commercial research/trial corpus license](https://www.comp.nus.edu.sg/~nlp/conll14st/nucle_license.pdf)
- **`DATA`** [Cornell Movie--Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- **`OFF-THE-SHELF`** [Pororo's GEC](https://kakaobrain.github.io/pororo/seq2seq/gec.html)
  
## Grapheme To Phoneme
  
- **`WIKI`** [Grapheme](https://en.wikipedia.org/wiki/Grapheme)  
- **`WIKI`** [Phoneme](https://en.wikipedia.org/wiki/Phoneme)  
- **`REPRESENTATIVE-DATA`** [Multilingual Pronunciation Data](https://drive.google.com/drive/folders/0B7R_gATfZJ2aWkpSWHpXUklWUmM)
- **`OFF-THE-SHELF-MODEL`** [Pororo's G2P](https://kakaobrain.github.io/pororo/seq2seq/g2p.html)
  
## Language Modeling
  
- **`WIKI`** [Language Model](https://en.wikipedia.org/wiki/Language_model)
- **`INFO`** [A beginner’s guide to language models](https://towardsdatascience.com/the-beginners-guide-to-language-models-aa47165b57f9)
- **`MODEL`** [GPT3](https://arxiv.org/abs/2005.14165)
- **`MODEL`** [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **`MODEL`** [Ken-LM](https://github.com/kpu/kenlm)
- **`MODEL`** [RNN-LM](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- **`CODE`** [Example with OpenWebText](https://github.com/sooftware/nlp-tasks/tree/main/language_modeling)
  
## Machine Reading Comprehension
  
- **`WIKI`** [Reading Comprehension](https://en.wikipedia.org/wiki/Reading_comprehension)
- **`INFO`** [Machine Reading Comprehension with BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/)
- **`DATA`** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- **`DATA`** [KorQuad](https://korquad.github.io/KorQuad%201.0/)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`MODEL`** [Electra](https://arxiv.org/abs/2003.10555)
- **`OFF-THE-SHELF`** [Pororo's MRC](https://kakaobrain.github.io/pororo/tagging/mrc.html)
- **`CODE`** [Example with SQuAD & KorQuad](https://github.com/sooftware/nlp-tasks/tree/main/machine_reading_comprehension)
  
## Machine Translation
  
- **`WIKI`** [Translation](https://en.wikipedia.org/wiki/Translation)
- **`DATA`** [WMT 2014 English-to-French](https://www.statmt.org/wmt14/translation-task.html)
- **`DATA`** [Korean-English translation corpus](https://aihub.or.kr/aidata/87)
- **`MODEL`** [Transformer](https://arxiv.org/abs/1706.03762)
- **`OFF-THE-SHELF`** [Pororo's Translation](https://kakaobrain.github.io/pororo/seq2seq/mt.html)
- **`CODE`** [Example with Korean-English translation corpus](https://github.com/sooftware/nlp-tasks/tree/main/machine_tranlsation)
  
## Math Word Problem Solving

- **`PAPER-WITH-CODE`** [Math Word Problem Solving](https://paperswithcode.com/task/math-word-problem-solving)
- **`DATA`** [DeepMind Mathmatics Dataset](https://github.com/deepmind/mathematics_dataset)
  
## Natural Language Inference
  
- **`WIKI`** [Textual Entailment](https://en.wikipedia.org/wiki/Textual_entailment)
- **`DATA`** [GLUE-MNLI](https://arxiv.org/abs/1804.07461)
- **`DATA`** [KorNLI](https://github.com/kakaobrain/KorNLUDatasets)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`MODEL`** [Electra](https://arxiv.org/abs/2003.10555)
- **`OFF-THE-SHELF`** [Pororo's NLI](https://kakaobrain.github.io/pororo/text_cls/nli.html)
- **`CODE`** [Example with GLUE-MNLI](https://github.com/sooftware/nlp-tasks/tree/main/natural_language_inference)
  
## Named Entity Recognition
  
- **`WIKI`** [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
- **`DATA`** [CoNLL-2002 NER corpus](https://github.com/teropa/nlp/tree/master/resources/corpora/conll2002)
- **`DATA`** [CoNLL-2003 NER corpus](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)
- **`DATA`** [Naver NER](https://github.com/naver/nlp-challenge)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`MODEL`** [Electra](https://arxiv.org/abs/2003.10555)
- **`OFF-THE-SHELF`** [Pororo's NER](https://kakaobrain.github.io/pororo/tagging/ner.html)
- **`CODE`** [Example with Naver NER](https://github.com/sooftware/nlp-tasks/tree/main/named_entity_recognition)

## Paraphrase Generation
  
- **`WIKI`** [Paraphrase](https://en.wikipedia.org/wiki/Paraphrase)  
- **`OFF-THE-SHELF`** [Pororo's Paraphrase Generation](https://kakaobrain.github.io/pororo/seq2seq/para_gen.html)
  
## Phoneme To Grapheme

- **`OFF-THE-SHELF`** [Pororo's P2G](https://kakaobrain.github.io/pororo/seq2seq/p2g.html)
  
## Sentiment Analysis
  
- **`WIKI`** [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
- **`DATA`** [GLUE-SST](https://arxiv.org/abs/1804.07461)
- **`DATA`** [NSMC](https://github.com/e9t/nsmc)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`MODEL`** [Electra](https://arxiv.org/abs/2003.10555)
- **`OFF-THE-SHELF`** [Pororo's Sentiment Analysis](https://kakaobrain.github.io/pororo/text_cls/sentiment.html)
- **`CODE`** [Example with NSMC](https://github.com/sooftware/nlp-tasks/tree/main/sentiment_classification)
  
## Semantic Textual Similarity
  
- **`WIKI`** [Semantic Similarity](https://en.wikipedia.org/wiki/Semantic_similarity)
- **`DATA`** [GLUE-STS](https://arxiv.org/abs/1804.07461)
- **`DATA`** [KorSTS](https://github.com/kakaobrain/KorNLUDatasets)
- **`MODEL`** [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- **`MODEL`** [RoBERTa](https://arxiv.org/abs/1907.11692)
- **`MODEL`** [Electra](https://arxiv.org/abs/2003.10555)
- **`OFF-THE-SHELF`** [Pororo's STS](https://kakaobrain.github.io/pororo/text_cls/sts.html)
- **`CODE`** [Example with SQuAD](https://github.com/sooftware/nlp-tasks/tree/main/semantic_textual_similarity)
  
## Speech Synthesis
  
- **`WIKI`** [Speech Synthesis](https://en.wikipedia.org/wiki/Speech_synthesis)
- **`DATA`** [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
- **`DATA`** [CSS10](https://github.com/Kyubyong/css10)
- **`DATA`** [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)
- **`MODEL`** [Tacotron2](https://arxiv.org/abs/1712.05884)
- **`MODEL`** [FastSpeech2](https://arxiv.org/abs/2006.04558)
- **`MODEL`** [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
- **`MODEL`** [Hifi-GAN](https://arxiv.org/abs/2010.05646)
- **`OFF-THE-SHELF`** [Pororo's TTS](https://kakaobrain.github.io/pororo/miscs/tts.html)
- **`CODE`** [Example with LJ-Speech](https://github.com/NVIDIA/tacotron2)
- **`CODE`** [Example with KSS](https://github.com/sooftware/taKotron2)
  
## Summarization
  
- **`WIKI`** [Automatic Summarization](https://en.wikipedia.org/wiki/Automatic_summarization)
- **`DATA`** [XSum](https://arxiv.org/abs/1808.08745)
- **`DATA`** [Korean Summarization Corpus](https://aihub.or.kr/aidata/8054)
- **`MODEL`** [BART](https://arxiv.org/abs/1910.13461)
- **`OFF-THE-SHELF`** [Pororo's Summarization](https://kakaobrain.github.io/pororo/seq2seq/summary.html)
- **`CODE`** [Example with XSum](https://github.com/sooftware/nlp-tasks/tree/main/summarization)
  
## Author

- [Soohwan Kim](https://github.com/sooftware) @sooftware
- Contacts: sh951011@gmail.com
