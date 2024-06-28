import os
import numpy as np
import pickle5 as pickle
import pickle5 as pickle
from scipy.ndimage import gaussian_filter
import pandas as pd

def smooth_logits_matrix(input_matrix, window_size=10):
    # Assuming input_matrix is a NumPy array of size Tx56

    smooth_matrix = np.zeros_like(input_matrix)

    for j in range(56):
        w = window_size  # Adjust the window size as needed

        all_logits = input_matrix[:, j]

        for k in range(len(all_logits)):
            if k < int(w / 2):
                all_logits[k] = all_logits[k]
            elif k > (len(all_logits) - int(w / 2)):
                all_logits[k] = all_logits[k]

        a = all_logits
        n = w

        one_new_logits = np.convolve(a, np.ones((n,))/n, mode='valid')
        smooth_matrix[int(w / 2): (len(all_logits) - int(w / 2)) + 1, j] = one_new_logits

    return smooth_matrix

prediction_path = './biwi/pred'
gt_path = '../data/CodeTalker/BIWI/emoca_biwi/'

out_pred = '../data/biwi_predictions/ours_pred7'
for pred_file in os.listdir(prediction_path):
    pred = np.load(os.path.join(prediction_path, pred_file))
    pred = smooth_logits_matrix(pred)
    # gt_filename: F5_e39_condition_F4.npy -> F5_e39.npy
    gt_filename = pred_file.split('.')[0] + '.pkl'
    gt = pickle.load(open(os.path.join(gt_path, gt_filename), 'rb'))
    frame_list = list(gt.keys())
    frame_list.sort()
    
    out_pred_file = os.path.join(out_pred, pred_file.split('.')[0])
    
    for frame_id in frame_list:
        try:
            frame_num = int(frame_id.split('_')[-1])

            exp_path_pred = os.path.join(out_pred_file, frame_id, 'exp.npy')
            pose_path_pred = os.path.join(out_pred_file, frame_id, 'pose.npy')
            cam_path_pred = os.path.join(out_pred_file, frame_id, 'cam.npy')
            shape_path_pred = os.path.join(out_pred_file, frame_id, 'shape.npy')

            # make dirs if not exist
            os.makedirs(os.path.dirname(exp_path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(pose_path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(cam_path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(shape_path_pred), exist_ok=True)

            # pose is zero, we don't want movement
            # cur_pose = np.zeros((1,6))
            frame_id = 'frame_'+str(frame_num).zfill(3)
            np.save(exp_path_pred, pred[frame_num-1][-50:])
            np.save(pose_path_pred, pred[frame_num-1][:6])
            np.save(cam_path_pred, gt[frame_id]['cam'])
            np.save(shape_path_pred, gt[frame_id]['shape'])
        except:
            continue

# out_path = '../data/BIWI_flames/emoca_all/'
# for gt_file in os.listdir(gt_path):
#     gt = pickle.load(open(os.path.join(gt_path, gt_file), 'rb'))
#     frame_list = list(gt.keys())
#     frame_list.sort()
#     file_id = gt_file.split('.')[0]
#     for frame_id in frame_list:
#         frame_num = int(frame_id.split('_')[-1])

#         exp_path_pred = os.path.join(out_path, file_id,  frame_id, 'exp.npy')
#         pose_path_pred = os.path.join(out_path, file_id, frame_id, 'pose.npy')
#         cam_path_pred = os.path.join(out_path, file_id, frame_id, 'cam.npy')
#         shape_path_pred = os.path.join(out_path, file_id, frame_id, 'shape.npy')

#         os.makedirs(os.path.dirname(exp_path_pred), exist_ok=True)
#         os.makedirs(os.path.dirname(pose_path_pred), exist_ok=True)
#         os.makedirs(os.path.dirname(cam_path_pred), exist_ok=True)
#         os.makedirs(os.path.dirname(shape_path_pred), exist_ok=True)

#         np.save(exp_path_pred, gt[frame_id]['exp'])
#         np.save(pose_path_pred, gt[frame_id]['pose'])
#         np.save(cam_path_pred, gt[frame_id]['cam'])
#         np.save(shape_path_pred, gt[frame_id]['shape'])