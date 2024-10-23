import os
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def softmax(logits):
    max_logits = torch.max(logits, dim=1, keepdim=True)[0] 
    logits_exp = torch.exp(logits - max_logits)
    sum_logits_exp = torch.sum(logits_exp, dim=1, keepdim=True)
    probs = logits_exp / sum_logits_exp
    return probs

def nll_loss(logits, labels):
    probs = softmax(logits)
    log_probs = torch.log(probs)
    nll = -log_probs.gather(dim=1, index=labels.unsqueeze(1))
    # loss = (-torch.log_softmax(logit, dim=1).select(dim=1, index=0)).mean()
    return nll.mean() # 각 배치에서 평균값 반환

def compute_loss(q_outputs, p_outputs, n_outputs):
    postitve_sim = torch.sum(q_outputs * p_outputs, dim=-1) # [B]
    negative_sim = torch.mm(q_outputs, n_outputs.T) # [B x B]
    logits = torch.cat([postitve_sim.unsqueeze(1), negative_sim], dim=1) # [B x (B+1)]
    labels = torch.zeros(q_outputs.size(0), dtype=torch.long).to(device) # 각 batch의 첫번째 위치에 정답이 있음
    loss = nll_loss(logits, labels)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    c_loss = cross_entropy_loss(logits, labels)
    return loss, c_loss

def train_DE(args, train_dataset, p_tokenizer, p_model, q_tokenizer, q_model):
    # Set encoders
    p_encoder = p_model.to(device)
    q_encoder = q_model.to(device)
    # Set model to train
    p_encoder.train()
    q_encoder.train()
    # Set parameters
    lr = args.lr
    batch_size = args.batch_size
    optimizer = optim.Adam(list(p_encoder.parameters())+list(q_encoder.parameters()), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            p_input_ids, p_attention_mask, q_input_ids, q_attention_mask, n_input_ids, n_attention_mask = batch['p_input_ids'].to(device), batch['p_attention_mask'].to(device), batch['q_input_ids'].to(device), batch['q_attention_mask'].to(device), batch['n_input_ids'].to(device), batch['n_attention_mask'].to(device)
            
            # [B x seq_len]
            q_input_ids = q_input_ids.view(-1, q_input_ids.size(-1))  # [B, list_len, seq_len] -> [B, seq_len]
            p_input_ids = p_input_ids.view(-1, p_input_ids.size(-1))  # [B x list_len x seq_len] -> [B x seq_len]
            n_input_ids = n_input_ids.view(-1, n_input_ids.size(-1))  # [B x list_len x seq_len] -> [B x seq_len]
            q_attention_mask = q_attention_mask.view(-1, q_attention_mask.size(-1))
            p_attention_mask = p_attention_mask.view(-1, p_attention_mask.size(-1))
            n_attention_mask = n_attention_mask.view(-1, n_attention_mask.size(-1))

            # q_outputs = q_encoder(q_input_ids, q_attention_mask)['hidden_states'][-1][:,0,:] # question
            # q_outputs = q_encoder(q_input_ids, q_attention_mask)['last_hidden_state'][:,0,:] # question
            q_outputs = q_encoder(q_input_ids, q_attention_mask).last_hidden_state[:,0,:] # question
            p_outputs = p_encoder(p_input_ids, p_attention_mask).last_hidden_state[:,0,:] # positive
            n_outputs = p_encoder(n_input_ids, n_attention_mask).last_hidden_state[:,0,:] # negative

            loss, cross_entropy_loss = compute_loss(q_outputs, p_outputs, n_outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss
        print(f"Training loss: {train_loss/len(train_dataloader)}")
        torch.save(p_encoder, os.path.join(args.home, args.model_dir, "p_model.pt"))
        torch.save(q_encoder, os.path.join(args.home, args.model_dir, "q_model.pt"))

    return p_tokenizer, p_model, q_tokenizer, q_model