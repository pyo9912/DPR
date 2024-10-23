import os
import pickle
from copy import deepcopy

def init_dir(default_args):
    args = deepcopy(default_args)
    args.home = os.path.dirname(os.path.dirname(__file__))
    return args

def load_pkl(args):
    train_pkl = pickle.load(open(os.path.join(args.home, "data", f"{args.train_pkl}.pkl"),"rb"))
    test_pkl = pickle.load(open(os.path.join(args.home, "data", f"{args.test_pkl}.pkl"), "rb"))
    return train_pkl, test_pkl

def load_knowledges(args):
    knowledge_pkl = pickle.load(open(os.path.join(args.home, "data", f"{args.knowledge_pkl}.pkl"),"rb"))
    return knowledge_pkl