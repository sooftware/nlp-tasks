# MIT License
# code by Soohwan Kim @sooftware

import librosa
import numpy as np
import logging
import torch
import os
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, extension: str = 'pcm'):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    try:
        if extension == 'pcm':
            signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
            return signal / 32767  # normalize audio

        elif extension == 'wav' or extension == 'flac':
            signal, _ = librosa.load(audio_path, sr=16000)
            return signal

    except ValueError:
        logger.debug('ValueError in {0}'.format(audio_path))
        return None
    except RuntimeError:
        logger.debug('RuntimeError in {0}'.format(audio_path))
        return None
    except IOError:
        logger.debug('IOError in {0}'.format(audio_path))
        return None


def collate_fn(batch, pad_token_id: int = 0):
    r"""
    Functions that pad to the maximum sequence length
    Args:
        batch (tuple): tuple contains input and target tensors
        pad_id (int): identification of pad token
    Returns:
        seqs (torch.FloatTensor): tensor contains input sequences.
        target (torch.IntTensor): tensor contains target sequences.
        seq_lengths (torch.IntTensor): tensor contains input sequence lengths
        target_lengths (torch.IntTensor): tensor contains target sequence lengths
    """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_token_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths


class SpeechToTextDataset(Dataset):
    def __init__(self, data_dir, audio_paths, transcripts, tokenizer, num_mels):
        super(SpeechToTextDataset, self).__init__()
        self.data_dir = data_dir
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.tokenizer = tokenizer
        self.sos_id = self.tokenizer.sos_id
        self.eos_id = self.tokenizer.eos_id
        self.num_mels = num_mels

    def _parse_audio(self, audio_path):
        signal = load_audio(audio_path, extension='pcm')
        mfcc = librosa.feature.mfcc(signal, sr=16000, n_mfcc=self.num_mels, n_fft=320, hop_length=160)

        mfcc -= mfcc.mean()
        mfcc /= np.std(mfcc)

        return torch.FloatTensor(mfcc).transpose(0, 1)

    def _parse_transcript(self, transcript: str) -> list:
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript

    def __getitem__(self, idx):
        mfcc = self._parse_audio(os.path.join(self.data_dir, self.audio_paths[idx]))
        transcript = self._parse_transcript(self.transcripts[idx])
        return mfcc, transcript

    def __len__(self):
        return len(self.audio_paths)
