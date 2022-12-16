# coding:utf-8

import torch
import os
from transformers import AdamW
from onestop_qamaker import OneStopQAMaker
from data_helper import QAGDataSet
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

Lambda = 0.5
batch_size = 100
epochs = 5
lr = 2e-5
vocab_size=21128
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = "./data/cmrc2018.json"
save_model_path = "./best_weight.pth"


def train(model, Lambda):
    train_dataset = QAGDataSet(data_path=data_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    criterion = CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        for step, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in data]
            true_start_id, true_end_id, true_decode_id = batch[4:]
            start_logits, end_logits, decoder_out = model(*batch[:4])

            true_decode_id = true_decode_id.view(-1)
            decoder_out = decoder_out.view(-1, vocab_size)

            loss_start_idx = criterion(start_logits, true_start_id)
            loss_end_idx = criterion(end_logits, true_end_id)
            loss_decoder_idx = criterion(decoder_out, true_decode_id)
            loss = Lambda * loss_decoder_idx + (1 - Lambda)*(loss_start_idx + loss_end_idx)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))

    torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    model = OneStopQAMaker()
    model.train()
    model.to(device)

    print("step1:  Fine-tune question generation .... ")
    Lambda = 1
    train(model, Lambda)

    print("step2:  Fine-tune answer prediction ..... ")
    Lambda = 0
    model.load_state_dict(torch.load(save_model_path))
    train(model, Lambda)

    print("step3: Fine-tune the OneStop model .... ")
    Lambda = 0.5
    model.load_state_dict(torch.load(save_model_path))
    train(model, Lambda)


