import os
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics import Perplexity
from tqdm import tqdm

def train_epoch(model, loader, optimizer, device, scheduler=None, clip=0.0, print_freq=100, epoch=0):
    model.train()
    losses = []
    total_tokens = 0
    # check if cuda is device 0 or not to decide print tqdm or not
    # for i, batch in enumerate(tqdm(loader, disable = device.index == 0)):
    for i, batch in enumerate(tqdm(loader)):
        src, tgt, src_len, (speaker_ids, listener_ids), data_ids = batch
        speaker_ids = speaker_ids.to(device)
        listener_ids = listener_ids.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
        for j in range(src.shape[0]):
            mask[j, :src_len[j]] = True
        optimizer.zero_grad()
        loss, _ = model(src, tgt, mask, speaker_ids=None, listener_ids=listener_ids)
        loss.mean().backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.mean().item())
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss_avg:.4f}\t'.format(
                   epoch, i, len(loader), loss_avg=np.mean(losses)))
            losses = []

def train_continuous_epoch(model, loader, optimizer, device, scheduler=None, clip=0.0, print_freq=100, epoch=0):
    model.train()
    losses = []
    for i, batch in enumerate(tqdm(loader)):
        src, tgt, src_len, _, _ = batch
        src = src.to(device)
        tgt = tgt.to(device)
        mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
        for j in range(src.shape[0]):
            mask[j, :src_len[j]] = True
        optimizer.zero_grad()
        loss = model(src, tgt, mask)
        loss.mean().backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.mean().item())
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss_avg:.4f}\t'.format(
                   epoch, i, len(loader), loss_avg=np.mean(losses)))
            losses = []


def evaluate_epoch(model, loader, device):
    # print loss, perplexity and bleu
    model.eval()
    losses = []
    perplexity_metric = Perplexity()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            src, tgt, src_len, _ = batch
            src = src.to(device)
            tgt = tgt.to(device)
            mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
            for j in range(src.shape[0]):
                mask[j, :src_len[j]] = True
            loss, logits = model(src, tgt, mask)
            z_pred, z_gt = model.module.generate(src, tgt, mask)
            for j in range(z_gt.shape[0]):
                perplexity_metric.update(logits[j][mask[j, 1:]].unsqueeze(0).cpu(), z_gt[j][1:][mask[j, 1:]].unsqueeze(0).cpu())
            losses.append(loss.mean().item())
    ppl = perplexity_metric.compute()
    # print loss and perplexity
    print('Validation: Loss {loss:.4f}\tPerplexity {perplexity:.3f}\t'.format(
        loss=np.mean(losses),
        perplexity=ppl))
    return ppl

def evaluate_continuous_epoch(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            src, tgt, src_len, _ = batch
            src = src.to(device)
            tgt = tgt.to(device)
            mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
            for j in range(src.shape[0]):
                mask[j, :src_len[j]] = True
            loss = model(src, tgt, mask)
            losses.append(loss.mean().item())
    # print loss 
    print('Validation: Loss {loss:.4f}\t'.format(
        loss=np.mean(losses)))
    return np.mean(losses)
