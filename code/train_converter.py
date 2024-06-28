import os
import torch
import torch.nn as nn
from dataset.biwi import get_dataloaders_convert
from seq2seq_pretrain import EmocaConverter

from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import builtins
import pickle5 as pickle

with open("../data/CodeTalker/BIWI/regions/lve.txt") as f:
    maps = f.read().split(", ")
    mouth_map = [int(i) for i in maps]

def train_epoch(model, loader, optimizer, device, scheduler=None, clip=0.0, print_freq=2000, epoch=0):
    model.train()
    losses = []
    mse_loss = nn.MSELoss()
    for i, batch in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        xv, xt, xe, _ = batch
        xv = xv.to(device)
        xt = xt.to(device)
        xe = xe.to(device)
        xp, _ = model(xv, xt, xe)
        xp_mouth = xp.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)
        xv_mouth = xv.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)

        # xp_mouth2 = xp2.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)
        # xv_mouth2 = xv.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)

        loss = mse_loss(xp, xv) + 5*mse_loss(xp_mouth, xv_mouth)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.mean().item())
        # if i % print_freq == 0:
    print('Epoch: [{0}][{1}/{2}]\t'
            'Loss {loss_avg:.4f}\t'.format(
            epoch, i, len(loader), loss_avg=np.mean(losses)))
    losses = []

def evaluate_epoch(model, loader, device):
    # print loss, perplexity and bleu
    model.eval()
    losses = []
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            xv, xt, xe, _ = batch
            xv = xv.to(device)
            xt = xt.to(device)
            xe = xe.to(device)
            xp, _ = model(xv, xt, xe)
            
            xp_mouth = xp.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)
            xv_mouth = xv.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)

            # xp_mouth2 = xp2.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)
            # xv_mouth2 = xv.reshape(1, -1, 23370, 3)[:, :, mouth_map, :].reshape(1, -1, 4996*3)

            loss = mse_loss(xp, xv) + 5*mse_loss(xp_mouth, xv_mouth)
            # loss = mse_loss(xp, xv)
            losses.append(loss.item())
    return np.mean(losses)


crank = 0
device = torch.device("cuda:{}".format(crank))
model = EmocaConverter().to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = None

# dataset = get_vico_dataloaders(batch_size=8)
dataset = get_dataloaders_convert(batch_size=1)
train_loader = dataset['train']
val_loader = dataset['train']

num_epochs = 1000
best_ppl = 10000
print(f'training for {num_epochs} epochs')
for epoch in range(num_epochs):
    model.train()
    train_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=1.0, print_freq=2000, epoch=epoch)
    val_loss = evaluate_epoch(model, val_loader, device)
    print(f'Epoch {epoch} val loss: {val_loss}')
    if val_loss < best_ppl:
        best_ppl = val_loss
        torch.save(model.state_dict(), 'best_converter.pt')
