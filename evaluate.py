import os
import glob
import pickle
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .u_models import SimCLR, SegmentationHead
from .u_eval import evaluate_model
from .u_datasets import read_data, OxfordPetsDataset
from .u_transformations import trans_config

def eval_each_method():
    """
    Returns:
        results_dict (dict): dictionary with key names model paths and values evlauation results (which are also dictionaries)
    """
    model_paths = np.concatenate(
        (['main_models/finished_model',
        'main_models/benchmark_model'],
        glob.glob('mrp_first_experiment_models/finetuned*'),
        glob.glob('oeq_models/finetuned*')
        ))
    
    model_paths = [f.replace('\\', '/') for f in model_paths]
    
    test_data = read_data('data/oxford/annotations/test.txt')
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_dict = {}

    for file_name in model_paths:
        t, _ = trans_config(0)
        oxford_test_dataset = OxfordPetsDataset('data/oxford', test_data, transform=t)
        oxford_test_dataloader = DataLoader(oxford_test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                
        model = SimCLR(out_features=128).to(device)
        model.head = SegmentationHead(in_features=512, output_dim=3)
        path = 'main_models/finished_model.pth'
        model.load_state_dict(torch.load(path))
        model.eval()

        results = evaluate_model(model, oxford_test_dataloader, device)

        results_dict[file_name] = results

    return results_dict

    