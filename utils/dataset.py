import random
import torch
import torch.utils.data as data

class CRSDataset(data.Dataset):
    def __init__(self, args, dataset, knowledges, p_tokenizer, q_tokenizer, max_len):
        self.args = args
        self.dataset = dataset
        self.knowledges = knowledges
        self.p_tokenizer = p_tokenizer
        self.q_tokenizer = q_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        context_batch = dict()

        dialog = [self.dataset[idx]['dialog']]
        target_knowledge = self.dataset[idx]['target_knowledge']
        target_idx = self.knowledges.index(target_knowledge)
        candidate_knowledges = self.dataset[idx]['candidate_knowledges']
        candidate_confidences = self.dataset[idx]['candidate_confidences']

        positive_idx = random.randint(0,self.args.n_positive-1)
        positive = candidate_knowledges[positive_idx]
        hard_negative_idx = random.randint(self.args.n_positive, self.args.n_hard-1)
        hard_negative = candidate_knowledges[hard_negative_idx]

        tokenized_question = self.q_tokenizer(dialog, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        tokenized_positive = self.p_tokenizer(positive, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        tokenized_negative = self.p_tokenizer(hard_negative, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        tokenized_target = self.p_tokenizer(target_knowledge, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        context_batch['q_input_ids'] = torch.tensor(tokenized_question['input_ids'])
        context_batch['q_attention_mask'] = torch.tensor(tokenized_question['attention_mask'])
        context_batch['p_input_ids'] = torch.tensor(tokenized_positive['input_ids'])
        context_batch['p_attention_mask'] = torch.tensor(tokenized_positive['attention_mask'])
        context_batch['n_input_ids'] = torch.tensor(tokenized_negative['input_ids'])
        context_batch['n_attention_mask'] = torch.tensor(tokenized_negative['attention_mask'])
        context_batch['t_input_ids'] = torch.tensor(tokenized_target['input_ids'])
        context_batch['labels'] = torch.tensor(target_idx)
        return context_batch