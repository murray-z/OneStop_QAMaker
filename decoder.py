import torch
import os
from onestop_qamaker import OneStopQAMaker
from transformers import BertTokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pretrained_model = "fnlp/bart-base-chinese"
save_model_path = "./best_weight.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_question_length = 32
max_encoder_length = 128
model = OneStopQAMaker()
model.load_state_dict(torch.load(save_model_path))
model.to(device)

tokenizer = BertTokenizer.from_pretrained(pretrained_model)
cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id

def decoder(input_text):
    encoder_inputs = tokenizer.encode_plus(input_text,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=max_encoder_length)

    encoder_input_ids = encoder_inputs["input_ids"].to(device)
    encoder_attention_mask = encoder_inputs["attention_mask"].to(device)

    decoder_input_ids = torch.tensor([[cls_id]], dtype=torch.long).to(device)
    decoder_attention_mask = torch.tensor([[1]], dtype=torch.long).to(device)

    question_ids = []

    for i in range(max_question_length):
        start_logits, end_logits, decoder_out = model(encoder_input_ids,
                                                      encoder_attention_mask,
                                                      decoder_input_ids,
                                                      decoder_attention_mask)

        values, indices = torch.topk(decoder_out, 1, dim=2)
        indice = indices[0, -1, -1].item()
        question_ids.append(indice)

        decoder_input_ids = torch.cat((decoder_input_ids, indices[:, -1, :]), dim=1).to(device)
        decoder_attention_mask = torch.cat((decoder_attention_mask, torch.tensor([[1]], device=device)), dim=1)

        if indice == sep_id or i == max_question_length-1:
            start_idx = torch.argmax(start_logits, dim=1)[0].item()
            end_idx = torch.argmax(end_logits, dim=1)[0].item()
            answer = input_text[start_idx-1:end_idx-1]
            question = tokenizer.decode(question_ids, skip_special_tokens=True)
            return {"question": question, "answer": answer}


if __name__ == '__main__':
    text = "中国的首都是北京。"
    res = decoder(text)
    print(res)




