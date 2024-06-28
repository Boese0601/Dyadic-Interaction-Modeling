import os
import numpy as np
from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance, calculate_variance, calcuate_sid, sts
from tqdm import tqdm
# import pdb


lm_listener_train_data_path = "../data/eccv2024/baselines/learning2listen/src/data/trevor/train"
lm_listener_test_data_path = "../data/eccv2024/baselines/learning2listen/src/data/trevor/test"
train_data_listener_data = np.load(os.path.join(lm_listener_train_data_path,"p0_list_faces_clean_deca.npy"))
train_data_speaker_data = np.load(os.path.join(lm_listener_train_data_path,"p1_speak_faces_clean_deca.npy"))
train_data_audio_data = np.load(os.path.join(lm_listener_train_data_path,"p1_speak_audio_clean_deca.npy"))

test_data_listener_data = np.load(os.path.join(lm_listener_test_data_path,"p0_list_faces_clean_deca.npy"))
test_data_speaker_data = np.load(os.path.join(lm_listener_test_data_path,"p1_speak_faces_clean_deca.npy"))
test_data_audio_data = np.load(os.path.join(lm_listener_test_data_path,"p1_speak_audio_clean_deca.npy"))
test_data_id_data = np.load(os.path.join(lm_listener_test_data_path,"p1_speak_files_clean_deca.npy"))

# pdb.set_trace()
audio_X, speaker_X, listener_X = [], [], []

for i in range(len(train_data_listener_data)):
    cur_audio = train_data_audio_data[i]
    cur_speaker = train_data_speaker_data[i]
    cur_listener = train_data_listener_data[i][:,:56]
    audio_X.append(cur_audio)
    speaker_X.append(cur_speaker)
    listener_X.append(cur_listener)

# nearest neighbor - audio
pred_dict, gt_dict, x_dict = {}, {}, {}
# for i in tqdm(range(15)):
# for i in tqdm(range(len(test_data_listener_data))):
#     cur_audio = test_data_audio_data[i]
#     cur_id = test_data_id_data[i][0][0]
#     cur_speaker = test_data_speaker_data[i][:,:56]
#     cur_listener = test_data_listener_data[i][:,:56]
#     # retrieve nearest neighbor
#     cur_vector = np.array(cur_audio).max(axis=0)
#     best_index, best_sim = 0, -1
#     # consine similarity
#     for j in range(len(audio_X)):
#         cur_sim = np.dot(cur_vector, audio_X[j].max(axis=0)) / (np.linalg.norm(cur_vector) * np.linalg.norm(audio_X[j].max(axis=0)))
#         if cur_sim > best_sim:
#             best_sim = cur_sim
#             best_index = j
#     pred_seq = listener_X[best_index]
#     if cur_id not in x_dict.keys():
#         pred_dict[cur_id] = []
#         gt_dict[cur_id] = []
#         x_dict[cur_id] = []
#         # pdb.set_trace() 
#     if len(pred_seq) == len(cur_listener) == len(cur_speaker) == 64:
#         pred_dict[cur_id].append(pred_seq)
#         gt_dict[cur_id].append(cur_listener)
#         x_dict[cur_id].append(cur_speaker)

 
# nearest neighbor - motion
# pred_dict, gt_dict, x_dict = {}, {}, {} 
# for i in tqdm(range(len(test_data_listener_data))):
#     cur_audio = test_data_audio_data[i]
#     cur_id = test_data_id_data[i][0][0]
#     cur_speaker = test_data_speaker_data[i][:,:56]
#     cur_listener = test_data_listener_data[i][:,:56]
#     # retrieve nearest neighbor
#     cur_vector = np.array(cur_speaker).mean(axis=0)
#     best_index, best_sim = 0, -1
#     # mse sim
#     for j in range(len(speaker_X)):
#         cur_sim = np.dot(cur_vector, speaker_X[j].mean(axis=0)) / (np.linalg.norm(cur_vector) * np.linalg.norm(speaker_X[j].mean(axis=0)))
#         if cur_sim > best_sim:
#             best_sim = cur_sim
#             best_index = j
#     pred_seq = listener_X[best_index]
#     if cur_id not in x_dict.keys():
#         pred_dict[cur_id] = []
#         gt_dict[cur_id] = []
#         x_dict[cur_id] = []
#     pred_dict[cur_id].append(pred_seq)
#     gt_dict[cur_id].append(cur_listener)
#     x_dict[cur_id].append(cur_speaker)

# random
# pdb.set_trace()
pred_dict, gt_dict, x_dict = {}, {}, {}
for i in tqdm(range(len(test_data_listener_data))):
    cur_audio = test_data_audio_data[i]
    cur_id = test_data_id_data[i][0][0]
    cur_speaker = test_data_speaker_data[i][:,:56]
    cur_listener = test_data_listener_data[i][:,:56]
    # randomly select best index
    # pred_seq = listener_X[np.random.randint(0, len(listener_X))]
    pred_seq = listener_X[np.random.randint(0, 5)]
    # pred_seq = cur_speaker
    if cur_id not in x_dict.keys():
        pred_dict[cur_id] = []
        gt_dict[cur_id] = []
        x_dict[cur_id] = []
    if len(pred_seq) == len(cur_listener) == len(cur_speaker) == 64:
        pred_dict[cur_id].append(pred_seq)
        gt_dict[cur_id].append(cur_listener)
        x_dict[cur_id].append(cur_speaker)
    
pred, gt, x = [], [], []
for key in pred_dict.keys():
    # concatenate all sequences, then append
    try:
        pred.append(np.concatenate(pred_dict[key], axis=0))
        gt.append(np.concatenate(gt_dict[key], axis=0))
        x.append(np.concatenate(x_dict[key], axis=0))
    except:
        continue


fids = []
for i in range(len(gt)):
    mu1, sigma1 = calculate_activation_statistics(gt[i][:, 50:])
    mu2, sigma2 = calculate_activation_statistics(pred[i][:, 50:])
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    fids.append(fid)
print('fid_pose: ', np.mean(fids))

fids = []
for i in range(len(gt)):
    mu1, sigma1 = calculate_activation_statistics(gt[i][:, :50])
    mu2, sigma2 = calculate_activation_statistics(pred[i][:, :50])
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    fids.append(fid)
print('fid_exp: ', np.mean(fids))

# pdb.set_trace()

pfids = []
for i in range(len(gt)):
    gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i][:,50:], gt[i][:,50:]], axis=-1))
    mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i][:,50:], pred[i][:,50:]], axis=-1))
    fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
    pfids.append(fid2)
print('pfid_pose: ', np.mean(pfids))

pfids = []
for i in range(len(gt)):
    gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i][:,:50], gt[i][:,:50]], axis=-1))
    mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i][:,:50], pred[i][:,:50]], axis=-1))
    fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
    pfids.append(fid2)
print('pfid_exp: ', np.mean(pfids))

total_l2 = []
for i in range(len(gt)):
    mse = np.mean((gt[i][:, 50:] - pred[i][:, 50:])**2)
    total_l2.append(mse)
print('mse_pose: ', np.mean(total_l2))

total_l2 = []
for i in range(len(gt)):
    mse = np.mean((gt[i][:, :50] - pred[i][:, :50])**2)
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
print('var_pose: ', np.var(gt[:, 50:].reshape(-1, )), np.var(pred[:, 50:].reshape(-1, )))
print('var_exp: ', np.var(gt[:, :50].reshape(-1, )), np.var(pred[:, :50].reshape(-1, )))

x = np.concatenate(x, axis=0)
x = x[:, 0:56]
pcc_xy_pose = np.corrcoef(gt[:, 50:].reshape(-1, ), x[:, 50:].reshape(-1, ))[0, 1]
pcc_xy_exp = np.corrcoef(gt[:, :50].reshape(-1, ), x[:, :50].reshape(-1, ))[0, 1]
pcc_xypred_pose = np.corrcoef(pred[:, 50:].reshape(-1, ), x[:, 50:].reshape(-1, ))[0, 1]
pcc_xypred_exp = np.corrcoef(pred[:, :50].reshape(-1, ), x[:, :50].reshape(-1, ))[0, 1]
# print('pcc pose: ', pcc_xy_pose, pcc_xypred_pose)
# print('pcc exp: ', pcc_xy_exp, pcc_xypred_exp)
print('rpcc pose: ', abs(pcc_xy_pose-pcc_xypred_pose))
print('rpcc exp: ', abs(pcc_xy_exp-pcc_xypred_exp))


sts_pose = sts(gt[:, 50:], pred[:, 50:])
sts_exp = sts(gt[:, :50], pred[:, :50])
print('sts pose: ', sts_pose)
print('sts exp: ', sts_exp)
print('here')