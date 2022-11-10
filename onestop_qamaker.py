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
        self.start_w = torch.nn.Parameter(data=torch.randn((embed_dim, embed_dim)), requires_grad=True)
        self.end_w = torch.nn.Parameter(data=torch.randn((embed_dim, embed_dim)), requires_grad=True)
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

        attention_out, attention_weight = self.attention(q, k, v)
        attention_out = torch.transpose(attention_out, 1, 2)

        start_logits = torch.matmul(encoder_last_hidden_state, self.start_w)
        start_logits = torch.bmm(start_logits, attention_out).squeeze(dim=-1)

        end_logits = torch.matmul(encoder_last_hidden_state, self.end_w)
        end_logits = torch.bmm(end_logits, attention_out).squeeze(dim=-1)

        decoder_out = self.decoder_out(decoder_last_hidden_state)

        return start_logits, end_logits, decoder_out





if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    encoder_inputs = tokenizer("中国的首都是北京市", return_tensors="pt")
    decoder_inputs = tokenizer("中国是首都是哪里", return_tensors="pt")

    encoder_input_ids = encoder_inputs["input_ids"]
    encoder_attention_mask = encoder_inputs["attention_mask"]
    decoder_input_ids = decoder_inputs["input_ids"]
    decoder_attention_mask = decoder_inputs["attention_mask"]


    qag = OneStopQAMaker()

    qag.forward(encoder_input_ids, encoder_attention_mask,
                decoder_input_ids, decoder_attention_mask)
