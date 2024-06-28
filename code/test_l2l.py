import os
import torch
import torch.nn as nn
from tqdm import tqdm
from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance, calculate_variance, calcuate_sid, sts
import numpy as np
import pickle5 as pickle
import pandas as pd

predicted_file = '../data/l2l_vico_predictions.pkl'
predicted_data = pickle.load(open(predicted_file, 'rb'))

vico_emoca = '../data/vico_dataset/emoca/'
gt_data = []
pred_data = []
xs = []

metadata_path = '../data/RLD_data.csv'

metadata = pd.read_csv(metadata_path).values
speaker_listener_dict = {}
for row in metadata:
    file_id = row[1]
    listener_file = row[2]
    speaker_file = row[3]
    speaker_listener_dict[file_id] = (speaker_file, listener_file)


for file_id in predicted_data.keys():
    try:
        cur_speaker_file, cur_listener_file = speaker_listener_dict[file_id]
    except:
        print(f'erron on {file_id}')
        continue
    
    cur_file_emoca_path = os.path.join(vico_emoca, cur_speaker_file, 'EMOCA_v2_lr_mse_20')
    all_frame_data = list(sorted(os.listdir(cur_file_emoca_path)))
    video_feats = []
    idx = 0
    for frame_data in all_frame_data:
        if frame_data.startswith('0'):
            # if idx >= target_len:
            #     break
            cur_frame_data_path = os.path.join(cur_file_emoca_path, frame_data)
            # if idx % 3 == 0:
            cur_frame_data_exp_path = os.path.join(cur_frame_data_path, 'exp.npy')
            cur_frame_data_pose_path = os.path.join(cur_frame_data_path, 'pose.npy')
            cur_frame_data_detail_path = os.path.join(cur_frame_data_path, 'detail.npy')

            cur_exp_data = np.load(cur_frame_data_exp_path)
            cur_pose_data = np.load(cur_frame_data_pose_path)
            cur_detail_data = np.load(cur_frame_data_detail_path)
            cur_frame_data = np.concatenate([cur_pose_data, cur_exp_data], axis=0)
            video_feats.append(cur_frame_data)
            idx += 1
    video_feats = np.array(video_feats)

    cur_file_emoca_path = os.path.join(vico_emoca, cur_listener_file, 'EMOCA_v2_lr_mse_20')
    all_frame_data = list(sorted(os.listdir(cur_file_emoca_path)))
    video_feats_speaker = []
    idx = 0
    for frame_data in all_frame_data:
        if frame_data.startswith('0'):
            # if idx >= target_len:
            #     break
            cur_frame_data_path = os.path.join(cur_file_emoca_path, frame_data)
            # if idx % 3 == 0:
            cur_frame_data_exp_path = os.path.join(cur_frame_data_path, 'exp.npy')
            cur_frame_data_pose_path = os.path.join(cur_frame_data_path, 'pose.npy')
            cur_frame_data_detail_path = os.path.join(cur_frame_data_path, 'detail.npy')

            cur_exp_data = np.load(cur_frame_data_exp_path)
            cur_pose_data = np.load(cur_frame_data_pose_path)
            cur_detail_data = np.load(cur_frame_data_detail_path)
            cur_frame_data = np.concatenate([cur_pose_data, cur_exp_data], axis=0)
            video_feats_speaker.append(cur_frame_data)
            idx += 1
    video_feats_speaker = np.array(video_feats_speaker)
    # append zeros to match target length
    predicted_seq = predicted_data[file_id]
    # swap 0:50, 50:56 to 0:6, 6:56
    predicted_seq = np.concatenate([predicted_seq[:, 50:56], predicted_seq[:, 0:50]], axis=1)
    
    target_len = min(len(video_feats), len(video_feats_speaker), len(predicted_seq))
    
    # predicted_seq = np.concatenate([predicted_seq, np.zeros((target_len-predicted_seq.shape[0], predicted_seq.shape[1]))], axis=0)
    # # convert all to float
    video_feats = video_feats.astype(np.float32)
    video_feats_speaker = video_feats_speaker.astype(np.float32)
    predicted_seq = predicted_seq.astype(np.float32)
    # predicted_seq: 0:50:56 -> 0:6:56
    # predicted_seq = np.concatenate([predicted_seq[:, 50:56], predicted_seq[:, 0:50]], axis=1)

    xs.append(video_feats_speaker[:target_len, :])
    gt_data.append(video_feats[:target_len, :])
    pred_data.append(predicted_seq)

import random

# replicate data 3 times each to match test legnth of over 30
expected_length = 43
gt, pred = [], []
x = []
for i in range(len(gt_data)):
    gt.append(gt_data[i])
    pred.append(pred_data[i])
    x.append(xs[i])

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

pfids = []
for i in range(len(gt)):
    gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i][:,0:6], gt[i][:,0:6]], axis=-1))
    mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i][:,0:6], pred[i][:,0:6]], axis=-1))
    fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
    pfids.append(fid2)
print('pfid_pose: ', np.mean(pfids))

pfids = []
for i in range(len(gt)):
    gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i][:,6:], gt[i][:,6:]], axis=-1))
    mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i][:,6:], pred[i][:,6:]], axis=-1))
    fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
    pfids.append(fid2)
print('pfid_exp: ', np.mean(pfids))

total_l2 = []
for i in range(len(gt)):
    mse = np.mean((gt[i][:, 0:6] - pred[i][:, 0:6])**2)
    total_l2.append(mse)
print('mse_pose: ', np.mean(total_l2))

total_l2 = []
for i in range(len(gt)):
    mse = np.mean((gt[i][:, 6:] - pred[i][:, 6:])**2)
    total_l2.append(mse)
print('mse_exp: ', np.mean(total_l2))

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