import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

hit = [1, 2, 3, 4, 5]
hit_scores = [0, 0, 0, 0, 0]

def test_DE(args, test_dataset, knowledges, passage_embs, p_tokenizer, p_model, q_tokenizer, q_model):
    with torch.no_grad():
        # Set encoders
        q_encoder = q_model.to(device)
        p_encoder = p_model.to(device)
        # Set model to eval
        q_encoder.eval()
        p_encoder.eval()

        batch_size = args.eval_batch_size
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for batch in tqdm(test_dataloader):
            q_input_ids, q_attention_mask, labels = batch['q_input_ids'].to(device), batch['q_attention_mask'].to(device), batch['labels'].to(device)

            # [B, seq_len]
            q_input_ids = q_input_ids.view(-1, q_input_ids.size(-1))
            q_attention_mask = q_attention_mask.view(-1, q_attention_mask.size(-1))
            q_outputs = q_encoder(q_input_ids, q_attention_mask).last_hidden_state[:, 0, :] # question

            # sim_scores = torch.matmul(q_outputs, torch.transpose(passage_embs.to(device),0,1))
            passage_embs = passage_embs.to(device)
            sim_scores = torch.mm(q_outputs, passage_embs.T) # passages are already encoded [B, all_knowledge_len]
            ranks = torch.argsort(sim_scores, dim=1, descending=True).squeeze() # [B, all_knowledge_len]

            for idx, i in enumerate(hit):
                ranks_tmp = ranks[:,:i] # [B, k (top-k의 k)]
                labels_tmp = labels.unsqueeze(1) # [B, 1]
                cnt = torch.sum(labels_tmp == ranks_tmp) # True인 애들만 더해줌
                hit_scores[idx] += cnt.tolist()

        for i in range(len(hit_scores)):
            hit_scores[i] /= len(test_dataset)
        
        print(hit_scores)

    return