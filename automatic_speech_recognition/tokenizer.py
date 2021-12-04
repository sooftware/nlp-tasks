# MIT License
# code by Soohwan Kim @sooftware

import csv


class KsponSpeechTokenizer:
    def __init__(self, vocab_path, pad_token, sos_token, eos_token, blank_token):
        super(KsponSpeechTokenizer, self).__init__()
        self.vocab_dict, self.id_dict = self.load_vocab(
            vocab_path=vocab_path,
            encoding='utf-8-sig',
        )
        self.labels = self.vocab_dict.keys()
        self.sos_id = int(self.vocab_dict[sos_token])
        self.eos_id = int(self.vocab_dict[eos_token])
        self.pad_id = int(self.vocab_dict[pad_token])
        self.blank_id = int(self.vocab_dict[blank_token])
        self.vocab_path = vocab_path

    def __len__(self):
        return len(self.labels)

    def decode(self, labels):
        r"""
        Converts label to string (number => Hangeul)
        Args:
            labels (numpy.ndarray): number label
        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def encode(self, sentence):
        label = str()

        for ch in sentence:
            try:
                label += (str(self.vocab_dict[ch]) + ' ')
            except KeyError:
                continue

        return label[:-1]

    def load_vocab(self, vocab_path, encoding='utf-8'):
        r"""
        Provides char2id, id2char
        Args:
            vocab_path (str): csv file with character labels
            encoding (str): encoding method
        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()

        try:
            with open(vocab_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(vocab_path))

    def __call__(self, sentence):
        return self.encode(sentence)
