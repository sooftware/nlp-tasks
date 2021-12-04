# MIT License
# code by Soohwan Kim @sooftware

from torch.utils.data import Dataset
from tqdm import tqdm


class FillMaskDataset(Dataset):
    def __init__(self, datas, tokenizer):
        super(FillMaskDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_ids = list()
        self.attention_masks = list()
        self.labels = list()

        for data in tqdm(datas, desc='Tokenizing..'):
            encoding_dict = self.tokenizer(data["text"], truncation=True, max_length=tokenizer.model_max_length)
            self.input_ids.append(encoding_dict['input_ids'])
            self.attention_masks.append(encoding_dict['attention_mask'])
            self.labels.append(encoding_dict['input_ids'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
