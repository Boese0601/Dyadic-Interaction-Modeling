import os
import torch
import torch.nn as nn
from dataset.data_loader import get_candor_dataloaders, get_vico_dataloaders, get_lm_listener_dataloaders
from x_engine_pt import train_epoch, evaluate_epoch
from seq2seq_pretrain import SLM

from tqdm import tqdm
from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance, calculate_variance, calcuate_sid, sts
import numpy as np
from piq import FID
import torch.distributed as dist
import builtins
import pickle5 as pickle

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_dist():
    env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }

    ngpus_per_node = torch.cuda.device_count()

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="nccl")
    crank = int(env_dict['RANK'])

    if(crank!=0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print(f'Init succesfully rank {crank}')
    return crank

# crank = initialize_dist()
crank = 0
device = torch.device("cuda:{}".format(crank))
model = SLM().to(device)
model = nn.DataParallel(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = None

# dataset = get_vico_dataloaders(batch_size=8)
dataset = get_candor_dataloaders(batch_size=32)
# dataset = get_lm_listener_dataloaders(batch_size=32)
train_loader = dataset['train']
val_loader = dataset['valid']

num_epochs = 100
best_ppl = 10000
print(f'training for {num_epochs} epochs')
for epoch in range(num_epochs):
    model.train()
    train_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=1.0, print_freq=2000, epoch=epoch)
    val_loss = evaluate_epoch(model, val_loader, device)
    print(f'Epoch {epoch} val loss: {val_loss}')
    if val_loss < best_ppl:
        best_ppl = val_loss
        torch.save(model.state_dict(), 'best_model_candor_pretrain_15.pt')
