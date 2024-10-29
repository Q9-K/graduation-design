import os
import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, help="configuration file", required=True)

args = parser.parse_args()
