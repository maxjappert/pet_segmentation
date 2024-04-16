import os
import glob
import pickle
import random
import re
import numpy as np
import sys
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
from PIL import Image

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.u_models import SimCLR, SegmentationHead
from utils.u_eval import evaluate_model
from utils.u_datasets import read_data, OxfordPetsDataset

def eval_each_method():
    """
    Returns:
        results_dict (dict): dictionary with key names model paths and values evlauation results (which are also dictionaries)
    """
    model_paths = np.concatenate(
        (['main_models/finished_model.pth',
        'main_models/benchmark.pth'],
        glob.glob('mrp_first_experiment_models/finetuned*'),
        glob.glob('oeq_models/finetuned*'),
        # glob.glob('mrp_first_experiment_models/benchmark*')
        ))

    model_paths = [f.replace('\\', '/') for f in model_paths]

    test_data = read_data('data/oxford/annotations/test.txt')
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dict = {}
    n = len(model_paths)

    for i, file_name in enumerate(model_paths):
        print(f'({i+1}/{n}) Evaluating model {file_name}\n')

        start = time.time()

        t = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images to 256x256
            transforms.ToTensor(),  # Convert the PIL Image to a tensor
        ])

        oxford_test_dataset = OxfordPetsDataset('data/oxford', test_data, transform=t)
        oxford_test_dataloader = DataLoader(oxford_test_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)

        model = SimCLR(out_features=128).to(device)
        model.head = SegmentationHead(in_features=512, output_dim=3)
        model.load_state_dict(torch.load(file_name, map_location=device))
        model.flatten = False
        model.to(device)
        model.eval()

        results = evaluate_model(model, oxford_test_dataloader, device)

        end = time.time()

        print(f'Results: (took {(end-start)/60}m)\n')
        print(results)

        results_dict[file_name] = results

    return results_dict

if __name__ == '__main__':
    results = eval_each_method()

    with open('results/eval_results_benchmark.pkl', 'wb') as f:
        pickle.dump(results, f)
