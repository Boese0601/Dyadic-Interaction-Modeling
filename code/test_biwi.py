import os
import torch
import torch.nn as nn
from dataset.data_loader import get_candor_dataloaders, get_vico_dataloaders, get_lm_listener_dataloaders
from dataset.biwi import get_dataloaders    
from x_engine_pt import train_epoch, evaluate_finetune_epoch, evaluate_test_epoch, evaluate_test_epoch_biwi
from seq2seq_pretrain import SLMFT, SpeakerSLMFT

from tqdm import tqdm
import numpy as np
from piq import FID
import torch.distributed as dist
import builtins
import pickle5 as pickle
from mymetrics import print_biwi_metrics
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
# model = SLMFT().to(device)
model = SpeakerSLMFT().to(device)
# model = nn.DataParallel(model)
d = torch.load('best_model_biwi_finetune1.pt')
model.load_state_dict(d)


dataset = get_dataloaders(batch_size=1)
# dataset = get_lm_listener_dataloaders(batch_size=32)
test_loader = dataset['valid']

num_epochs = 100
best_ppl = 10000
# print(f'training for {num_epochs} epochs')
# for epoch in range(num_epochs):
#     model.train()
#     train_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=1.0, print_freq=10, epoch=epoch)

#     y_true, y_pred, x, data_ids = evaluate_finetune_epoch(model, val_loader, device)
#     fid_pose, fid_exp = print_metrics(y_true, y_pred, x)
#     if fid_pose + fid_exp < best_ppl:
#         best_ppl = fid_pose + fid_exp
#         torch.save(model.state_dict(), 'best_model_vico_finetune.pt')
# print(best_ppl)

y_true, y_pred, x, data_ids = evaluate_test_epoch_biwi(model, test_loader, device, beam_size=50)
gt_save_path = 'biwi/gt/'
pred_save_path = 'biwi/pred/'
# check if the folder exists and if not, create it
if not os.path.exists(gt_save_path):
    os.makedirs(gt_save_path)
if not os.path.exists(pred_save_path):
    os.makedirs(pred_save_path)
for idx, data_id in enumerate(data_ids):
    data_id = data_id.split('.')[0]
    np.save(gt_save_path + data_id+'.npy', y_true[idx])
    np.save(pred_save_path + data_id+'.npy', y_pred[idx])
