import os
import torch
import torch.nn as nn
from dataset.data_loader import get_candor_dataloaders, get_vico_dataloaders
from x_engine import train_epoch, evaluate_epoch
from seq2seq import ListenerGenerator, ContinuousTransformer
from tqdm import tqdm
from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance, calculate_variance, calcuate_sid, sts
import numpy as np
from piq import FID

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ListenerGenerator().to(device)
# model = ContinuousTransformer(
#     dim_in=768+56,
#     dim=512,
#     enc_depth = 6,
#     enc_heads = 8,
#     enc_max_seq_len = 1024,
#     dec_depth = 6,
#     dec_heads = 8,
#     dec_max_seq_len = 1024,
# ).to(device)

model = nn.DataParallel(model, device_ids=[0,1,2])
pretrained_ckpt = 'saved_checkpoints_vico/best_model30fps_ftvq.pth'
# pretrained_ckpt = 'saved_checkpoints_continuous_vico/best_model.pth'
# pretrained_ckpt = 'saved_checkpoints_vico/best_model.pth'
model.load_state_dict(torch.load(pretrained_ckpt))
model = model.module
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = None

dataset = get_vico_dataloaders(batch_size=1)
val_loader = dataset['valid']

model.eval()
gt, pred = [], []
x = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(val_loader)):
        src, tgt, src_len, _ = batch
        src = src.to(device)
        tgt = tgt.to(device)
        mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
        for j in range(src.shape[0]):
            mask[j, :src_len[j]] = True
        loss, logits = model(src, tgt, mask)
        z_pred, z_gt = model.generate(src, tgt, mask)
        for j in range(z_gt.shape[0]):
            pred_seq = logits[j][mask[j, 1:]].unsqueeze(0).cpu()
            pred_seq = torch.argmax(pred_seq, dim=-1)
            # pred_seq = z_pred[j][mask[j]].cpu()
            pred_seq = pred_seq.squeeze().unsqueeze(1)
            min_encodings = torch.zeros(pred_seq.shape[0], 512)
            min_encodings.scatter_(1, pred_seq, 1)
            zq = torch.matmul(min_encodings.cuda(), model.listener_vq.quantize.embedding.weight)
            zq = zq.unsqueeze(0)
            zq = zq.permute(0, 2, 1)
            pred_cont_seq = model.listener_vq.decode(zq)[0].cpu().numpy()
            # gt_seq = tgt[j][1:][mask[j, 1:]].cpu().numpy()
            gt_seq = tgt[j][mask[j]][1:].cpu().numpy()
            pred.append(pred_cont_seq)
            gt.append(gt_seq)
            x.append(src[j][1:][mask[j, 1:]].cpu().numpy())

# with torch.no_grad():
#     for i, batch in enumerate(tqdm(val_loader)):
#         src, tgt, src_len, _ = batch
#         src = src.to(device)
#         tgt = tgt.to(device)
#         mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
#         for j in range(src.shape[0]):
#             mask[j, :src_len[j]] = True 
#         pred_seq = model.generate(src, mask)
#         for j in range(src.shape[0]):
#             gt.append(tgt[j][mask[j]].cpu().numpy())
#             pred.append(pred_seq[j][mask[j]].squeeze().cpu().numpy())
#             x.append(src[j][mask[j]].cpu().numpy())

# compute evaluation metrics
# sid_pose = calcuate_sid(gt, pred, type='pose')
# print('sid_pose: ', sid_pose)

# sid_exp = calcuate_sid(gt, pred, type='exp')
# print('sid_exp: ', sid_exp)

# gt = np.concatenate(gt, axis=0)
# pred = np.concatenate(pred, axis=0)
# print(gt.shape, pred.shape)

# mu1, sigma1 = calculate_activation_statistics(gt)
# mu2, sigma2 = calculate_activation_statistics(pred)
# fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
# print('fid: ', fid)

# print('l1: ', np.mean(np.abs(gt - pred)))
# print('l2: ', np.mean(np.square(gt - pred)))
# print('var: ', calculate_variance(gt), calculate_variance(pred))

# fid = FID()(torch.from_numpy(gt), torch.from_numpy(pred))
# print('fid: ', fid)

# print('l2: ', np.mean(np.square(gt - pred)))
# print('l1: ', np.mean(np.abs(gt - pred)))

# mu1, sigma1 = calculate_activation_statistics(gt[:, 0:6])
# mu2, sigma2 = calculate_activation_statistics(pred[:, 0:6])
# fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
# print('fid_pose: ', fid)


# mu1, sigma1 = calculate_activation_statistics(gt[:, 6:])
# mu2, sigma2 = calculate_activation_statistics(pred[:, 6:])
# fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
# print('fid_exp: ', fid)


# print('var_pose: ', calculate_variance(gt[:, 0:6]), calculate_variance(pred[:, 0:6]))
# print('var_exp: ', calculate_variance(gt[:, 6:]), calculate_variance(pred[:, 6:]))

print(len(gt))

fids = []
for i in range(len(gt)):
    mu1, sigma1 = calculate_activation_statistics(gt[i][:, 0:6])
    mu2, sigma2 = calculate_activation_statistics(pred[i][:, 0:6])
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    fids.append(fid)
print('fid_pose: ', np.mean(fids))

fids = []
for i in range(len(gt)):
    mu1, sigma1 = calculate_activation_statistics(gt[i][:, 6:])
    mu2, sigma2 = calculate_activation_statistics(pred[i][:, 6:])
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    fids.append(fid)
print('fid_exp: ', np.mean(fids))

sid_pose = calcuate_sid(gt, pred, type='pose')
sid_pose_gt = calcuate_sid(gt, gt, type='pose')
print('sid_pose: ', sid_pose, sid_pose_gt)

sid_exp = calcuate_sid(gt, pred, type='exp')
sid_exp_gt = calcuate_sid(gt, gt, type='exp')
print('sid_exp: ', sid_exp, sid_exp_gt)

gt = np.concatenate(gt, axis=0)
gt = gt.reshape(-1, 56)
pred = np.concatenate(pred, axis=0)
pred = pred.reshape(-1, 56)
print('var_pose: ', np.var(gt[:, 0:6].reshape(-1, )), np.var(pred[:, 0:6].reshape(-1, )))
print('var_exp: ', np.var(gt[:, 6:].reshape(-1, )), np.var(pred[:, 6:].reshape(-1, )))

x = np.concatenate(x, axis=0)
x = x[:, 0:56]
pcc_xy_pose = np.corrcoef(gt[:, 0:6].reshape(-1, ), x[:, 0:6].reshape(-1, ))[0, 1]
pcc_xy_exp = np.corrcoef(gt[:, 6:].reshape(-1, ), x[:, 6:].reshape(-1, ))[0, 1]
pcc_xypred_pose = np.corrcoef(pred[:, 0:6].reshape(-1, ), x[:, 0:6].reshape(-1, ))[0, 1]
pcc_xypred_exp = np.corrcoef(pred[:, 6:].reshape(-1, ), x[:, 6:].reshape(-1, ))[0, 1]
# print('pcc pose: ', pcc_xy_pose, pcc_xypred_pose)
# print('pcc exp: ', pcc_xy_exp, pcc_xypred_exp)
print('rpcc pose: ', abs(pcc_xy_pose-pcc_xypred_pose))
print('rpcc exp: ', abs(pcc_xy_exp-pcc_xypred_exp))

sts_pose = sts(gt[:, 0:6], pred[:, 0:6])
sts_exp = sts(gt[:, 6:], pred[:, 6:])
print('sts pose: ', sts_pose)
print('sts exp: ', sts_exp)
