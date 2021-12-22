import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

#instance = "John was born in New York. Where did John marry?"

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = GPT2LMHeadModel.from_pretrained('gpt2')

class qa_model(nn.Module):
    def __init__(self, LM):
        super(qa_model, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(LM)
        self.R = 10
        self.L = 20
        self.K = 10
        self.epsilon = 1e-6
        
    def compute_logprob(self, input_ids, attention_mask):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        logits = outputs[0][:, -1, :]
        logprob = F.log_softmax(logits, dim=-1)
        return torch.index_select(logprob, 1, input_ids.squeeze()) / attention_mask
    
    def candidate_spans(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        logprob = torch.log(torch.zeros([batch, self.R, self.L], dtype = torch.float32))
        logprob_first_token = self.compute_logprob(input_ids, attention_mask) #.squeeze()
        first_token_index = [torch.topk(logprob_first_token[i], self.R).indices.squeeze() for i in range(batch)]
        for b in range(batch):
            for i, index in enumerate(first_token_index[b]):
                for l in range(self.L - 1):
                    if index+l+1 < input_ids[b].squeeze().shape[0]:
                        if l == 0:
                            logprob[b][i][l] = logprob_first_token[b][index]
                        tmp = input_ids[b].squeeze()[index:index+l+1] # range does not include right index
                        tmp_input_ids = torch.cat((input_ids[b][:torch.sum(attention_mask[b]).int()].view(1, -1), tmp.view(1, -1)), dim=1)
                        tmp_attention_mask = torch.Tensor([1.0] * tmp_input_ids.shape[-1]).to(attention_mask.device)
                        logprob_tmp = self.compute_logprob(tmp_input_ids, tmp_attention_mask).squeeze()
                        logprob[b][i][l+1] = logprob_tmp[index+l+1] + logprob[b][i][l]

        topk_logprob = [torch.topk(logprob[b].flatten(), self.K) for b in range(batch)]
        topk_span_ids = [topk_logprob[b].indices for b in range(batch)]
        batch_spans = []
        for b in range(batch):
            spans = []
            for i in topk_span_ids[b]:
                start = first_token_index[int(i/self.L)]
                length = i % self.L
                spans.append([start, start+length+1])
            batch_spans.append(spans)
        return [topk_logprob[b].values for b in range(batch)], batch_spans
    
    def forward(self, input_ids, input_mask, answers):
        batch = input_ids.shape[0]
        probs, preds = self.candidate_spans(input_ids, input_mask)
        loss = 0.0
        topk = 0
        top1 = 0
        for b in range(batch):
            if len(answers[b]) > 0:
                target_prob = torch.tensor([1.0] * self.K)
                loss += F.kl_div(F.log_softmax(probs[b]), F.softmax(target_prob), reduction="none").mean()
            else:
                target_prob = torch.tensor([self.epsilon] * self.K)
                correct = []
                for k in range(self.K):
                    if preds[b][k][0] == answers[b][0] and preds[b][k][1] == answers[b][1]:
                        correct.append(k)
                        topk += 1
                        if torch.argmax(probs[b]) == k:
                            top1 += 1
                if correct:
                    for k in correct:
                        target_prob[k] = (1.0 - (self.K - len(correct)) * self.epsilon) / len(correct)
                        loss += F.kl_div(F.log_softmax(probs[b]), F.softmax(target_prob), reduction="none").mean()
                
                
        return probs, preds, topk, top1, loss        
    