import time
import datetime
import random
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from model import qa_model
from exp import *

# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
set_seed(42)

def collate_fn(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [50256] * (max_len - len(f['input_ids'])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    answers = [f['answers'] for f in batch]
    output = (input_ids, input_mask, answers)
    return output
    
def featurization(split):
    dataset = load_dataset('squad_v2', split=split)
    features = []
    for d in dataset:
        answers = set()
        if len(d['answers']['text']) > 0:
            for i in range(len(d['answers']['text'])):
                length = len(tokenizer.encode(d['answers']['text'][i]))
                start = d['answers']['answer_start'][i]
                answers.add((start, start + length))
        feature = {'input_ids': tokenizer.encode(d['context'] + "\n" + d['question']), \
                   'answers': answers
                  }
        features.append(feature)
    return features

#train_dataloader = DataLoader(featurization('train'), batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True)
valid_dataloader = DataLoader(featurization('validation'), batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
train_dataloader = test_dataloader = valid_dataloader

#############################
### Setting up parameters ###
#############################

params = {'transformers_model': 'gpt2',
          'batch_size': 1,    
          'epochs': 1,
          'learning_rate': 1e-6,
          'seed': 42,
          'gpu_id': '0',    
          'rst_file_name': "1222.rst"
         }

os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_id']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cuda = torch.device('cuda')

model = qa_model(params['transformers_model'])
model.to(cuda)
model.zero_grad()

print("# of parameters:", count_parameters(model))
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())
        
model_params_dir = "./model_params/"
rst_file_name = params['rst_file_name']
best_PATH = model_params_dir + rst_file_name.replace(".rst", ".pt") # to save model params here
model_name = rst_file_name.replace(".rst", "")

print("batch_size:", params['batch_size'])
total_steps = len(train_dataloader) * params['epochs']
print("Total steps: [number of batches] x [number of epochs] = " + str(len(train_dataloader)) + "x" + str(params['epochs']) + " = " + str(total_steps))

my_exp = exp(cuda, model, params['epochs'], params['learning_rate'], train_dataloader, valid_dataloader, test_dataloader, best_PATH, None, model_name)
perf = my_exp.train()

# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)
