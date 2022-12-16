# coding:utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from torch.nn import MultiheadAttention

from transformers import BartTokenizer, BartModel, BertTokenizer


class OneStopQAMaker(nn.Module):
    def __init__(self,
                 model_name="fnlp/bart-base-chinese",
                 embed_dim=768,
                 num_heads=12,
                 vocab_size=21128):
        super(OneStopQAMaker, self).__init__()
        self.bart = BartModel.from_pretrained(model_name)
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc_start = nn.Linear(embed_dim, embed_dim)
        self.fc_end = nn.Linear(embed_dim, embed_dim)
        self.vocab_size = vocab_size
        self.decoder_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_input_ids, encoder_attention_mask,
                decoder_input_ids, decoder_attention_mask):

        res = self.bart(encoder_input_ids, encoder_attention_mask,
                        decoder_input_ids, decoder_attention_mask,
                        return_dict=True)

        decoder_last_hidden_state = res["last_hidden_state"]
        encoder_last_hidden_state = res["encoder_last_hidden_state"]

        q = decoder_last_hidden_state[:,-1,:]
        q = torch.unsqueeze(q, 1)
        k = v = encoder_last_hidden_state

        q = torch.transpose(q, 0, 1)
        k = torch.transpose(k, 0, 1)
        v = torch.transpose(v, 0, 1)

        attention_out, attention_weight = self.attention(q, k, v)
        attention_out = attention_out.permute(1, 2, 0)

        start_logits = self.fc_start(encoder_last_hidden_state)
        start_logits = torch.bmm(start_logits, attention_out).squeeze(dim=-1)

        end_logits = self.fc_end(encoder_last_hidden_state)
        end_logits = torch.bmm(end_logits, attention_out).squeeze(dim=-1)

        decoder_out = self.decoder_out(decoder_last_hidden_state)

        return start_logits, end_logits, decoder_out

