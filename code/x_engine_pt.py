import os
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics import Perplexity
from tqdm import tqdm
from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance

def train_epoch(model, loader, optimizer, device, scheduler=None, clip=0.0, print_freq=2000, epoch=0):
    model.train()
    losses = []
    total_tokens = 0
    # check if cuda is device 0 or not to decide print tqdm or not
    # for i, batch in enumerate(tqdm(loader, disable = device.index == 0)):
    d = {
            'l_ce_s': 0,
            'l_ce_l': 0,
            'l_cont_s': 0,
            'l_cont_l': 0,
            'nce': 0,
            'c_acc': 0
        }
    for i, batch in enumerate(tqdm(loader)):
        src, tgt, src_len, _, data_ids = batch
        # speaker_ids = speaker_ids.to(device)
        # listener_ids = listener_ids.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        # split feature_dim of src into 56 and 768
        src_s_v, src_s_a = torch.split(src, [56, 768], dim=2)
        mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
        for j in range(src.shape[0]):
            mask[j, :src_len[j]] = True
        optimizer.zero_grad()
        loss, d_step, _ = model(src_s_v, tgt, src_s_a, mask, mode='train')
        loss.mean().backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for k in d:
            try:
                d[k] += d_step[k].mean().item()
            except:
                d[k] += 0
        losses.append(loss.mean().item())
        if i % print_freq == 0:
            for k in d:
                d[k] /= print_freq
            # print epoch along with d
            print('Epoch {epoch} Batch {i}:\tLoss {loss:.4f}\tCE_s {l_ce_s:.4f}\tCE_l {l_ce_l:.4f}\tCont_s {l_cont_s:.4f}\tCont_l {l_cont_l:.4f}\tNCE {nce:.4f}\tC_acc {c_acc:.4f}'.format(
                epoch=epoch,
                i=i,
                loss=np.mean(losses),
                **d))
            # reset d
            for k in d:
                d[k] = 0
            losses = []

def train_epoch_biwi(model, loader, optimizer, device, scheduler=None, clip=0.0, print_freq=2000, epoch=0):
    model.train()
    losses = []
    total_tokens = 0
    # check if cuda is device 0 or not to decide print tqdm or not
    # for i, batch in enumerate(tqdm(loader, disable = device.index == 0)):
    d = {
            'l_ce_s': 0,
            'l_ce_l': 0,
            'l_cont_s': 0,
            'l_cont_l': 0,
            'nce': 0,
            'c_acc': 0
        }
    mapper = {
        'F2': 0,
        'F3': 1,
        'F4': 2,
        'M3': 3,
        'M4': 4,
        'M5': 5,
        'F1': 6,
        'F5': 7,
        'F6': 8,
        'F7': 9,
        'F8': 10,
        'M1': 11,
        'M2': 12,
        'M6': 13
    }
    for i, batch in enumerate(tqdm(loader)):
        xa, xv, xt, xe, fnames = batch
        xa = xa.to(device)
        xv = xv.to(device)
        xt = xt.to(device)
        xe = xe.to(device)

        speaker_ids = []
        for fname in fnames:
            speaker_ids.append(mapper[fname.split('_')[0]])
        speaker_ids = torch.tensor(speaker_ids).long().to(device)
    
        mask = torch.ones((xa.shape[0], xa.shape[1]), dtype=torch.bool).to(device)
        optimizer.zero_grad()
        # v_speaker, v_listener, v_audio, mask, template
        loss, d_step, _ = model(xv, xe, xa, mask, xt, speaker_ids=speaker_ids)
        loss.mean().backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for k in d:
            try:
                d[k] += d_step[k].mean().item()
            except:
                d[k] += 0
        losses.append(loss.mean().item())
        if i % print_freq == 0:
            for k in d:
                d[k] /= print_freq
            # print epoch along with d
            print('Epoch {epoch} Batch {i}:\tLoss {loss:.4f}\tCE_s {l_ce_s:.4f}\tCE_l {l_ce_l:.4f}\tCont_s {l_cont_s:.4f}\tCont_l {l_cont_l:.4f}\tNCE {nce:.4f}\tC_acc {c_acc:.4f}'.format(
                epoch=epoch,
                i=i,
                loss=np.mean(losses),
                **d))
            # reset d
            for k in d:
                d[k] = 0
            losses = []

def evaluate_epoch(model, loader, device):
    # print loss, perplexity and bleu
    model.eval()
    losses = []
    total_tokens = 0
    d = {
            'l_ce_s': 0,
            'l_ce_l': 0,
            'l_cont_s': 0,
            'l_cont_l': 0,
            'nce': 0,
            'c_acc': 0
        }
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            src, tgt, src_len, (speaker_ids, listener_ids), data_ids = batch
            speaker_ids = speaker_ids.to(device)
            listener_ids = listener_ids.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            src_s_v, src_s_a = torch.split(src, [56, 768], dim=2)
            mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
            for j in range(src.shape[0]):
                mask[j, :src_len[j]] = True
            loss, d_step, _ = model(src_s_v, tgt, src_s_a, mask)
            losses.append(loss.mean().item())
            for k in d:
                d[k] += d_step[k].mean().item()
        for k in d:
            d[k] /= len(loader)
    print(d)
    return np.mean(losses)

def evaluate_epoch_biwi(model, loader, device):
    # print loss, perplexity and bleu
    model.eval()
    losses = []
    total_tokens = 0
    d = {
            'l_ce_s': 0,
            'l_ce_l': 0,
            'l_cont_s': 0,
            'l_cont_l': 0,
            'nce': 0,
            'c_acc': 0
        }
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            xa, xv, xt, _, _ = batch
            xa = xa.to(device)
            xv = xv.to(device)
            xt = xt.to(device)
        
            mask = torch.ones((xa.shape[0], xa.shape[1]), dtype=torch.bool).to(device)
            
            loss, d_step, _ = model(xv, None, xa, mask, xt, mode='val')
            losses.append(loss.mean().item())
            for k in d:
                try:
                    d[k] += d_step[k].item()
                except:
                    d[k] += d_step[k]
        for k in d:
            d[k] /= len(loader)
    print(d)
    return np.mean(losses)

def evaluate_finetune_epoch(model, loader, device):
    y_trues_all, y_preds_all = [], []
    x_all = []
    data_ids_all = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            src, tgt, src_len, _, data_ids = batch
            # speaker_ids = speaker_ids.to(device)
            # listener_ids = listener_ids.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            src_s_v, src_s_a = torch.split(src, [56, 768], dim=2)
            mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
            for j in range(src.shape[0]):
                mask[j, :src_len[j]] = True
            loss, d_step, y_preds = model(src_s_v, tgt, src_s_a, mask, mode='train')
            y_true = tgt[:, 1: , :]
            for j in range(len(y_preds)):
                # extract according to src_len
                y_preds_all.append(y_preds[j][:src_len[j]-1, :].cpu().numpy())
                y_trues_all.append(y_true[j][:src_len[j]-1, :].cpu().numpy())
                x_all.append(src_s_v[j, :src_len[j]-1, :].cpu().numpy())

                # for speaker behavior prediction
                # y_preds_all.append(y_preds[j][:src_len[j]-1, :].cpu().numpy())
                # y_trues_all.append(src_s_v[j][:src_len[j]-1, :].cpu().numpy())
                # x_all.append(src_s_v[j, :src_len[j]-1, :].cpu().numpy())
                data_ids_all.append(data_ids[j])
    return y_trues_all, y_preds_all, x_all, data_ids_all

def evaluate_test_epoch(model, loader, device):
    y_trues_all, y_preds_all = [], []
    x_all = []
    data_ids_all = []
    beam_size = 10
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            src, tgt, src_len, _, data_ids = batch
            # speaker_ids = speaker_ids.to(device)
            # listener_ids = listener_ids.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            src_s_v, src_s_a = torch.split(src, [56, 768], dim=2)
            mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
            for j in range(src.shape[0]):
                mask[j, :src_len[j]] = True
            y_true = tgt[:, 1: , :]
            for j in range(src.shape[0]):
                y_trues_all.append(y_true[j][:src_len[j]-1, :].cpu().numpy())
                data_ids_all.append(data_ids[j])
                x_all.append(src_s_v[j, :src_len[j]-1, :].cpu().numpy())
            cur_best = [float('inf') for _ in range(src.shape[0])]
            y_preds_all_cur = [None for _ in range(src.shape[0])]
            for _ in range(beam_size):
                loss, d_step, y_preds = model(src_s_v, tgt, src_s_a, mask, mode='val')
                
                for j in range(src.shape[0]):
                    cp = y_preds[j][:src_len[j]-1, :].cpu().numpy()
                    ct = y_true[j][:src_len[j]-1, :].cpu().numpy()
                    mu1, sigma1 = calculate_activation_statistics(ct)
                    mu2, sigma2 = calculate_activation_statistics(cp)
                    cfid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                    if cfid < cur_best[j]:
                        y_preds_all_cur[j] = y_preds[j][:src_len[j]-1, :].cpu().numpy()
                        cur_best[j] = cfid
            for j in range(src.shape[0]):
                y_preds_all.append(y_preds_all_cur[j])
                
    # for speaker behavior prediction
    # y_preds_all.append(y_preds[j][:src_len[j]-1, :].cpu().numpy())
    # y_trues_all.append(src_s_v[j][:src_len[j]-1, :].cpu().numpy())
    # x_all.append(src_s_v[j, :src_len[j]-1, :].cpu().numpy())
                
    return y_trues_all, y_preds_all, x_all, data_ids_all

def evaluate_test_epoch_biwi(model, loader, device, beam_size = 10):
    y_trues_all, y_preds_all = [], []
    x_all = []
    data_ids_all = []
    # beam_size = 50
    model.eval()

    mapper = {
        'F2': 0,
        'F3': 1,
        'F4': 2,
        'M3': 3,
        'M4': 4,
        'M5': 5,
        'F1': 6,
        'F5': 7,
        'F6': 8,
        'F7': 9,
        'F8': 10,
        'M1': 11,
        'M2': 12,
        'M6': 13
    }

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            xa, xv, xt, xe, data_ids = batch
            xa = xa.to(device)
            xv = xv.to(device)
            xt = xt.to(device)
            xe = xe.to(device)
            y_true = xe[:, 1: , :]
            src = xv
            fnames = data_ids
            speaker_ids = []
            for fname in fnames:
                speaker_ids.append(mapper[fname.split('_')[0]])
            speaker_ids = torch.tensor(speaker_ids).long().to(device)
            mask = torch.ones((xa.shape[0], xa.shape[1]), dtype=torch.bool).to(device)
            # v_speaker, v_listener, v_audio, mask, template
            
            for j in range(src.shape[0]):
                y_trues_all.append(y_true[j, 1:, :].cpu().numpy())
                data_ids_all.append(data_ids[j])
            cur_best = [float('inf') for _ in range(src.shape[0])]
            y_preds_all_cur = [None for _ in range(src.shape[0])]
            for _ in range(beam_size):
                loss, d_step, y_preds = model(xv, xe, xa, mask, xt, mode='train', speaker_ids=speaker_ids)
                
                for j in range(src.shape[0]):
                    cp = y_preds[j, 1:, :].cpu().numpy()
                    ct = y_true[j, 1:, :].cpu().numpy()
                    distance = np.mean(np.sqrt(np.sum((cp - ct) ** 2, axis=1)))
                    if distance < cur_best[j]:
                        y_preds_all_cur[j] = y_preds[j, 1:, :].cpu().numpy()
                        cur_best[j] = distance
            for j in range(src.shape[0]):
                y_preds_all.append(y_preds_all_cur[j])
                
    return y_trues_all, y_preds_all, x_all, data_ids_all