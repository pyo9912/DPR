import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--home", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--model_dir", type=str, default="saved_models")
    # dataset
    parser.add_argument("--knowledge_pkl", type=str, default="KnowledgeDB_durec_all")
    parser.add_argument("--train_pkl", type=str, default="train_pred_aug_dataset_new")
    parser.add_argument("--test_pkl", type=str, default="test_pred_aug_dataset_new")
    # model
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    # global var
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--cutoff", type=int, default=256)
    # train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_positive", type=int, default=2)
    parser.add_argument("--n_hard", type=int, default=20)
    # test
    parser.add_argument("--eval_batch_size", type=int, default=4)

    args = parser.parse_args()
    return args
