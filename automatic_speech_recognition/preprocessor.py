# MIT License
# code by Soohwan Kim @sooftware

import re
import os
import pandas as pd
import logging
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

logger = logging.getLogger(__name__)


def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


PERCENT_FILES = {
    '087797': '퍼센트',
    '215401': '퍼센트',
    '284574': '퍼센트',
    '397184': '퍼센트',
    '501006': '프로',
    '502173': '프로',
    '542363': '프로',
    '581483': '퍼센트'
}


def read_preprocess_text_file(file_path, mode):
    with open(file_path, 'r', encoding='cp949') as f:
        raw_sentence = f.read()
        file_name = os.path.basename(file_path)
        if file_name[12:18] in PERCENT_FILES.keys():
            replace = PERCENT_FILES[file_name[12:18]]
        else:
            replace = None
        return sentence_filter(raw_sentence, mode=mode, replace=replace)


def preprocess(dataset_path, mode='phonetic'):
    print('preprocess started..')

    audio_paths = list()
    transcripts = list()

    with Parallel(n_jobs=cpu_count() - 1) as parallel:
        for folder in os.listdir(dataset_path):
            # folder : {KsponSpeech_01, ..., KsponSpeech_05}
            path = os.path.join(dataset_path, folder)
            if not folder.startswith('KsponSpeech') or not os.path.isdir(path):
                continue

            subfolders = os.listdir(path)
            for idx, subfolder in tqdm(list(enumerate(subfolders)), desc=f'Preprocess text files on {path}'):
                path = os.path.join(dataset_path, folder, subfolder)
                if not os.path.isdir(path):
                    continue

                # list-up files
                sub_file_list = []
                audio_sub_file_list = []
                for file_name in os.listdir(path):
                    if file_name.endswith('.txt'):
                        sub_file_list.append(os.path.join(path, file_name))
                        audio_sub_file_list.append(os.path.join(folder, subfolder, file_name))

                # do parallel and get results
                new_sentences = parallel(
                    delayed(read_preprocess_text_file)(p, mode) for p in sub_file_list
                )

                audio_paths.extend(audio_sub_file_list)
                transcripts.extend(new_sentences)

    return audio_paths, transcripts


def preprocess_test_data(manifest_file_dir: str, mode='phonetic'):
    audio_paths = list()
    transcripts = list()

    for split in ("eval_clean.trn", "eval_other.trn"):
        with open(os.path.join(manifest_file_dir, split), encoding='utf-8') as f:
            for line in f.readlines():
                audio_path, raw_transcript = line.split(" :: ")
                transcript = sentence_filter(raw_transcript, mode=mode)

                audio_paths.append(audio_path)
                transcripts.append(transcript)

    return audio_paths, transcripts


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]


def generate_character_labels(transcripts, labels_dest, vocab_size: int = 2000):
    logger.info('create_char_labels started..')

    label_list = list()
    label_freq = list()

    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_list:
                label_list.append(ch)
                label_freq.append(1)
            else:
                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2, 3], 'char': ['<pad>', '<sos>', '<eos>', '<blank>'], 'freq': [0, 0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 4)
        label['char'].append(ch)
        label['freq'].append(freq)

    label['id'] = label['id'][:vocab_size]
    label['char'] = label['char'][:vocab_size]
    label['freq'] = label['freq'][:vocab_size]

    label_df = pd.DataFrame(label)
    label_df.to_csv(labels_dest, encoding="utf-8", index=False)


def generate_character_script(audio_paths: list, transcripts: list, manifest_file_path: str, vocab_path: str):
    logger.info('create_script started..')
    char2id, id2char = load_label(vocab_path)

    with open(manifest_file_path, "w") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            char_id_transcript = sentence_to_target(transcript, char2id)
            audio_path = audio_path.replace('txt', 'pcm')
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')


class KsponSpeechPreprocessor:
    KSPONSPEECH_TRAIN_NUM = 620000
    KSPONSPEECH_VALID_NUM = 2545
    KSPONSPEECH_TEST_NUM = 6000

    def __init__(self, dataset_path, preprocess_mode, test_manifest_path, vocab_path, vocab_size):
        super(KsponSpeechPreprocessor, self).__init__()
        self.dataset_path = dataset_path
        self.preprocess_mode = preprocess_mode
        self.test_manifest_path = test_manifest_path
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size

    def _generate_manifest_files(self, manifest_file_path: str) -> None:
        r"""
        Generate KsponSpeech manifest file.
        Format: AUDIO_PATH [TAB] TEXT_TRANSCRIPTS [TAB] NUMBER_TRANSCRIPT
        """
        train_valid_audio_paths, train_valid_transcripts = preprocess(
            self.dataset_path, self.preprocess_mode
        )
        test_audio_paths, test_transcripts = preprocess_test_data(
            self.test_manifest_path, self.preprocess_mode
        )

        audio_paths = train_valid_audio_paths + test_audio_paths
        transcripts = train_valid_transcripts + test_transcripts

        generate_character_labels(transcripts, self.vocab_path, self.vocab_size)
        generate_character_script(audio_paths, transcripts, manifest_file_path, self.vocab_path)

    def _parse_manifest_file(self, manifest_file_path):
        r"""
        Parsing manifest file.

        Returns:
            audio_paths (list): list of audio path
            transcritps (list): list of transcript of audio
        """
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path, encoding='utf-8-sig') as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, korean_transcript, transcript = line.split('\t')
                transcript = transcript.replace('\n', '')

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def setup(self, manifest_file_path):
        if not os.path.exists(manifest_file_path):
            self._generate_manifest_files(manifest_file_path)
        audio_paths, transcripts = self._parse_manifest_file(manifest_file_path)

        valid_end_idx = self.KSPONSPEECH_TRAIN_NUM + self.KSPONSPEECH_VALID_NUM

        audio_paths = {
            "train": audio_paths[:self.KSPONSPEECH_TRAIN_NUM],
            "valid": audio_paths[self.KSPONSPEECH_TRAIN_NUM:valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        transcripts = {
            "train": transcripts[:self.KSPONSPEECH_TRAIN_NUM],
            "valid": transcripts[self.KSPONSPEECH_TRAIN_NUM:valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }
        return audio_paths, transcripts