import torch
import os
from onestop_qamaker import OneStopQAMaker
from transformers import BertTokenizer
import torch.nn.functional as F

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


def greedy_decode(input_text):
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


def predict(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
    start_logits, end_logits, decoder_out = model(encoder_input_ids,
                                                  encoder_attention_mask,
                                                  decoder_input_ids,
                                                  decoder_attention_mask)
    decoder_out = F.log_softmax(decoder_out[:, -1, :], dim=-1)
    return start_logits, end_logits, decoder_out


def beam_search_decode(input_text, topk=3, min_len=1, min_ends=1):
    encoder_inputs = tokenizer.encode_plus(input_text,
                                           return_tensors="pt",
                                           padding=True,
                                           truncation=True,
                                           max_length=max_encoder_length)

    encoder_input_ids = encoder_inputs["input_ids"].to(device)
    encoder_attention_mask = encoder_inputs["attention_mask"].to(device)

    decoder_input_ids = torch.tensor([[cls_id]], dtype=torch.long).to(device)
    decoder_attention_mask = torch.tensor([[1]], dtype=torch.long).to(device)

    output_scores = torch.tensor(1, dtype=torch.float).to(device)

    for i in range(max_question_length):
        start_logits, end_logits, scores = predict(encoder_input_ids,
                                                        encoder_attention_mask,
                                                        decoder_input_ids,
                                                        decoder_attention_mask)

        vocab_size = scores.shape[1]

        if i == 0:
            encoder_input_ids = encoder_input_ids[0].repeat(topk, 1)
            encoder_attention_mask = encoder_attention_mask[0].repeat(topk, 1)
            decoder_input_ids = decoder_input_ids[0].repeat(topk, 1)
            decoder_attention_mask = decoder_attention_mask[0].repeat(topk, 1)

        # 累计得分
        scores = output_scores.reshape((-1, 1)) + scores
        scores = scores.view(-1)
        values, indices = torch.topk(scores, topk)

        indices_1 = (indices // vocab_size)
        indices_2 = (indices % vocab_size).reshape((-1, 1))

        decoder_input_ids = torch.cat([decoder_input_ids[indices_1], indices_2], dim=1)
        decoder_attention_mask = torch.cat([decoder_attention_mask, torch.tensor([1]).repeat(topk, 1).to(device)], dim=1)

        # 更新得分
        output_scores = scores[indices]

        # 统计出现结束符号次数
        end_counts = torch.sum(decoder_input_ids == sep_id, dim=1)

        # 判断是否达到最短长度
        if decoder_input_ids.shape[1] >= min_len:
            best_one = torch.argmax(output_scores)
            # 最优路径已达到结束符号
            if end_counts[best_one] == min_ends:
                start_idx = torch.argmax(start_logits, dim=1)[0].item()
                end_idx = torch.argmax(end_logits, dim=1)[0].item()
                answer = input_text[start_idx - 1:end_idx - 1]
                question = tokenizer.decode(decoder_input_ids[best_one], skip_special_tokens=True)
                return {"question": question, "answer": answer}
            else:
                # 未达到结束符号序列
                flag = (end_counts < min_ends)
                # 有已完成序列，但是得分不是最高；删除已经完成序列
                if not flag.all():
                    encoder_input_ids = encoder_input_ids[flag]
                    encoder_attention_mask = encoder_attention_mask[flag]
                    decoder_input_ids = decoder_input_ids[flag]
                    decoder_attention_mask = decoder_attention_mask[flag]
                    output_scores = output_scores[flag]
                    topk = flag.sum()

    # 达到设置最长长度
    best_one = torch.argmax(output_scores)
    start_idx = torch.argmax(start_logits, dim=1)[0].item()
    end_idx = torch.argmax(end_logits, dim=1)[0].item()
    answer = input_text[start_idx - 1:end_idx - 1]
    question = tokenizer.decode(decoder_input_ids[best_one], skip_special_tokens=True)
    return {"question": question, "answer": answer}




if __name__ == '__main__':
    text = "中国的首都是北京."

    greedy_res = greedy_decode(text)

    beam_search_res = beam_search_decode(text)

    print("greedy res:\n", greedy_res)
    print("beam_search res:\n", greedy_res)





