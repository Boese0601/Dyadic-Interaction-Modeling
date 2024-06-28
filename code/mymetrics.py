from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance, calculate_variance, calcuate_sid, sts
import numpy as np
import argparse
import os
import pickle

def print_metrics(y_true, y_pred, x):
    gt = y_true
    pred = y_pred

    fids = []
    for i in range(len(gt)):
        mu1, sigma1 = calculate_activation_statistics(gt[i][:, 0:6])
        mu2, sigma2 = calculate_activation_statistics(pred[i][:, 0:6])
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fids.append(fid)
    print('fid_pose: ', np.mean(fids))
    fid_pose = np.mean(fids)

    fids = []
    for i in range(len(gt)):
        mu1, sigma1 = calculate_activation_statistics(gt[i][:, 6:])
        mu2, sigma2 = calculate_activation_statistics(pred[i][:, 6:])
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fids.append(fid)
    print('fid_exp: ', np.mean(fids))
    fid_exp = np.mean(fids)

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
    return fid_pose, fid_exp

def print_metrics_full(y_true, y_pred, x):
    gt = y_true
    pred = y_pred

    fids = []
    for i in range(len(gt)):
        mu1, sigma1 = calculate_activation_statistics(gt[i])
        mu2, sigma2 = calculate_activation_statistics(pred[i])
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fids.append(fid)
    print('fid: ', np.mean(fids))
    
    pfids = []
    for i in range(len(gt)):
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i], gt[i]], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i], pred[i]], axis=-1))
        fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        pfids.append(fid2)
    print('pfid: ', np.mean(pfids))

    total_l2 = []
    for i in range(len(gt)):
        mse = np.mean((gt[i] - pred[i])**2)
        total_l2.append(mse)
    print('mse: ', np.mean(total_l2))

    gt = np.concatenate(gt, axis=0)
    gt = gt.reshape(-1, 56)
    pred = np.concatenate(pred, axis=0)
    pred = pred.reshape(-1, 56)
    print('var: ', np.var(gt.reshape(-1, )), np.var(pred.reshape(-1, )))

def print_biwi_metrics(y_true, y_pred, file_names):
    templates_path = "../data/BIWI_data/templates.pkl"
    region_path = "../data/CodeTalker/BIWI/regions/"
    with open(templates_path, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    with open(os.path.join(region_path, "lve.txt")) as f:
        maps = f.read().split(", ")
        mouth_map = [int(i) for i in maps]

    with open(os.path.join(region_path, "fdd.txt")) as f:
        maps = f.read().split(", ")
        upper_map = [int(i) for i in maps]

    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []
    for i in range(len(y_true)):
        vertices_gt = y_true[i].reshape(-1,23370,3)
        vertices_pred = y_pred[i].reshape(-1,23370,3)
        subject = file_names[i].split("_")[0]

        vertices_pred = vertices_pred[:vertices_gt.shape[0],:,:]

        motion_pred = vertices_pred - templates[subject].reshape(1,23370,3)
        motion_gt = vertices_gt - templates[subject].reshape(1,23370,3)

        cnt += vertices_gt.shape[0]

        vertices_gt_all.extend(list(vertices_gt))
        vertices_pred_all.extend(list(vertices_pred))

        L2_dis_upper = np.array([np.square(motion_gt[:,v, :]) for v in upper_map])
        L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
        L2_dis_upper = np.sum(L2_dis_upper,axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        gt_motion_std = np.mean(L2_dis_upper)
        
        L2_dis_upper = np.array([np.square(motion_pred[:,v, :]) for v in upper_map])
        L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
        L2_dis_upper = np.sum(L2_dis_upper,axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        pred_motion_std = np.mean(L2_dis_upper)

        motion_std_difference.append(gt_motion_std - pred_motion_std)
    
    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)
    

    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:,v, :]-vertices_pred_all[:,v,:]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1,0,2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max,axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max,axis=1)

    print('Lip Vertex Error: {:.4e}'.format(np.mean(L2_dis_mouth_max)))
    print('FDD: {:.4e}'.format(sum(motion_std_difference)/len(motion_std_difference)))
    lve = np.mean(L2_dis_mouth_max)
    fdd = sum(motion_std_difference)/len(motion_std_difference)
    return lve, fdd
