import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertTokenizer
import json


class QAGDataSet(Dataset):
    def __init__(self,
                 data_path,
                 model_name="fnlp/bart-base-chinese",
                 max_encoder_len=128,
                 max_decoder_len=64):
        super(QAGDataSet, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        with open(data_path, encoding="utf-8") as f:
            datas = json.loads(f.read())

        self.datas = []
        for p, q, a in datas:
            if len(p)+2 <= max_encoder_len:
                self.datas.append((p, q, a))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        p, q, a = self.datas[idx]

        # print(p, q, a)
        start_idx = p.index(a) + 1
        end_idx = start_idx + len(a)
        encoder_inputs = self.tokenizer.encode_plus(p,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_encoder_len)

        decoder_inputs = self.tokenizer.encode_plus(q,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_decoder_len)

        encoder_input_ids = encoder_inputs["input_ids"][0]
        encoder_attention_mask = encoder_inputs["attention_mask"][0]

        decoder_input_ids = decoder_inputs["input_ids"][0][:-1]
        decoder_output_ids = decoder_inputs["input_ids"][0][1:]

        decoder_attention_mask = decoder_inputs["attention_mask"][0][:-1]

        start_idx = torch.tensor(start_idx, dtype=torch.long)
        end_idx = torch.tensor(end_idx, dtype=torch.long)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, \
               start_idx, end_idx, decoder_output_ids
