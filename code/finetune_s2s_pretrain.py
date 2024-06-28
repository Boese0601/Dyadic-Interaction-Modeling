import os
import torch
import torch.nn as nn
from dataset.data_loader import get_candor_dataloaders, get_vico_dataloaders
from dataset.biwi import get_dataloaders
from dataset.l2l import get_lm_listener_dataloaders
from x_engine_pt import train_epoch, evaluate_finetune_epoch, train_epoch_biwi, evaluate_epoch_biwi, evaluate_test_epoch_biwi 
from seq2seq_pretrain import SLMFT, SpeakerSLMFT

from tqdm import tqdm
import numpy as np
from piq import FID
import torch.distributed as dist
import builtins
import pickle5 as pickle
from mymetrics import print_biwi_metrics, print_metrics
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
mode = 'listener'
if mode == 'speaker':
    # crank = initialize_dist()
    crank = 0
    device = torch.device("cuda:{}".format(crank))
    # model = SLMFT().to(device)
    model = SpeakerSLMFT().to(device)
    model = nn.DataParallel(model)
    d = torch.load('best_model_candor_pretrain_fix_zs.pt')
    modified_d = {}
    for k, v in d.items():
        if 'gamma' in k:
            modified_d[k.replace('gamma', 'weight')] = v
        elif 'beta' in k:
            modified_d[k.replace('beta', 'bias')] = v
        else:
            modified_d[k] = v
    model.load_state_dict(modified_d, strict=False)
    model = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = None

    # dataset = get_vico_dataloaders(batch_size=4)
    # # dataset = get_lm_listener_dataloaders(batch_size=32)
    # train_loader = dataset['train']
    # val_loader = dataset['valid']

    dataset = get_dataloaders(batch_size=1)
    train_loader = dataset['train']
    val_loader = dataset['valid']

    num_epochs = 150
    best_ppl = 10000
    print(f'training for {num_epochs} epochs')
    for epoch in range(num_epochs):
        model.train()
        train_epoch_biwi(model, train_loader, optimizer, device, scheduler=scheduler, clip=1.0, print_freq=100, epoch=epoch)
        if epoch % 10 == 0:
            y_true, y_pred, x, data_ids = evaluate_test_epoch_biwi(model, val_loader, device, beam_size=2)
            # lve, _ = print_biwi_metrics(y_true, y_pred, data_ids)
            lve = np.mean([np.mean((y_true[i] - y_pred[i])**2) for i in range(len(y_true))])
            if lve < best_ppl:
                best_ppl = lve
                torch.save(model.state_dict(), 'best_model_biwi_finetune1.pt')
        # fid_pose, fid_exp = print_metrics(y_true, y_pred, x)
        # if fid_pose + fid_exp < best_ppl:
        #     best_ppl = fid_pose + fid_exp
            # torch.save(model.state_dict(), 'best_model_vico_finetune.pt')
    print(best_ppl)
    #TODO: retrain with motion predictor (loss over motion, not face mesh)

        # # save predictions
        # d = {
        #     'y_true': y_true,
        #     'y_pred': y_pred,
        #     'data_ids': data_ids
        # }
        # with open('speaker_predictions.pkl', 'wb') as f:
        #     pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # crank = initialize_dist()
    crank = 0
    device = torch.device("cuda:{}".format(crank))
    model = SLMFT().to(device)
    model = nn.DataParallel(model)
    # d = torch.load('best_model_candor_pretrain_fix_zs.pt')
    # modified_d = {}
    # for k, v in d.items():
    #     if 'gamma' in k:
    #         modified_d[k.replace('gamma', 'weight')] = v
    #     elif 'beta' in k:
    #         modified_d[k.replace('beta', 'bias')] = v
    #     else:
    #         modified_d[k] = v
    # model.load_state_dict(modified_d, strict=False)
    model = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = None

    dataset = get_vico_dataloaders(batch_size=4)
    # dataset = get_lm_listener_dataloaders(batch_size=16)
    # train_loader = dataset['train']
    # val_loader = dataset['valid']

    # dataset = get_dataloaders(batch_size=1)
    train_loader = dataset['train']
    val_loader = dataset['valid']

    num_epochs = 100
    best_ppl = 10000
    print(f'training for {num_epochs} epochs')
    for epoch in range(num_epochs):
        model.train()
        train_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=1.0, print_freq=100, epoch=epoch)
        if epoch % 1 == 0:
            y_true, y_pred, x, data_ids = evaluate_finetune_epoch(model, val_loader, device)
            # lve, _ = print_biwi_metrics(y_true, y_pred, data_ids)
            # lve = np.mean([np.mean((y_true[i] - y_pred[i])**2) for i in range(len(y_true))])
            a, b = print_metrics(y_true, y_pred, x)
            if a+b < best_ppl:
                best_ppl = a+b
                torch.save(model.state_dict(), 'best_vico_causal.pt')
        # fid_pose, fid_exp = print_metrics(y_true, y_pred, x)
        # if fid_pose + fid_exp < best_ppl:
        #     best_ppl = fid_pose + fid_exp
            # torch.save(model.state_dict(), 'best_model_vico_finetune.pt')
    print(best_ppl)
    #TODO: retrain with motion predictor (loss over motion, not face mesh)

        # # save predictions
        # d = {
        #     'y_true': y_true,
        #     'y_pred': y_pred,
        #     'data_ids': data_ids
        # }
        # with open('speaker_predictions.pkl', 'wb') as f:
        #     pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)