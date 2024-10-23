import os
import sys
import pickle
import torch

from utils.parser import parse_args
from utils.utils import init_dir, load_pkl, load_knowledges
from utils.dataset import CRSDataset
from tqdm import tqdm
from train import train_DE
from test import test_DE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def init_model(args):
    model_name = args.model_name
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True) # 모든 인코더 layer에서 임베딩을 얻을 수 있게함
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens = True
    return tokenizer, model

def passage_embedding(args, p_tokenizer, p_encoder, knowledges):
    passage_embs = []
    with torch.no_grad():
        p_encoder.to(device)
        p_encoder.eval()
        for p in tqdm(knowledges):
            p = p_tokenizer(p, padding='max_length', truncation=True, max_length=args.cutoff, return_tensors='pt')
            p_input_ids = p['input_ids'].to(device)
            p_attention_mask = p['attention_mask'].to(device)
            p_emb = p_encoder(p_input_ids, p_attention_mask).last_hidden_state[:,0,:] # [1, d]
            passage_embs.append(p_emb)
        passage_embs = torch.cat(passage_embs) # [M, d]
    return passage_embs

if __name__ == "__main__":
    args = parse_args()
    args = init_dir(args)

    p_tokenizer, p_model = init_model(args)
    q_tokenizer, q_model = init_model(args)
    # truncation side setting
    q_tokenizer.truncation_side = 'left'
    p_tokenizer.truncation_side = 'right'

    train_dataset, test_dataset = load_pkl(args)
    knowledges = load_knowledges(args)

    train_dataset = CRSDataset(args, train_dataset, knowledges, p_tokenizer=p_tokenizer, q_tokenizer=q_tokenizer, max_len=args.cutoff)
    test_dataset = CRSDataset(args, test_dataset, knowledges, p_tokenizer=p_tokenizer, q_tokenizer=q_tokenizer, max_len=args.cutoff)

    if args.mode == "train":
        p_tokenizer, p_model, q_tokenizer, q_model = train_DE(args, train_dataset, p_tokenizer, p_model, q_tokenizer, q_model)
    elif args.mode == "test":
        p_model = torch.load(os.path.join(args.home, args.model_dir, "p_model.pt"))
        q_model = torch.load(os.path.join(args.home, args.model_dir, "q_model.pt"))
        passage_embs = passage_embedding(args, p_tokenizer, p_model, knowledges)
        test_DE(args, test_dataset, knowledges, passage_embs, p_tokenizer, p_model, q_tokenizer, q_model)
    else:
        print("Check mode")
        pass

    print()
