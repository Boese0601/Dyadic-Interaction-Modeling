import os
import pickle5 as pickle
import numpy as np
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

metadata_path = '../data/RLD_data.csv'

metadata = pd.read_csv(metadata_path).values
speaker_listener_dict = {}
for row in metadata:
    file_id = row[1]
    listener_file = row[2]
    speaker_file = row[3]
    speaker_listener_dict[file_id] = (speaker_file, listener_file)

# Load data from output.pkl
with open('listener_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

# Define the output directory
output_dir_pred = 'output_data_listener_new'
output_dir_gt = 'output_data_listener_new_gt'
os.makedirs(output_dir_pred, exist_ok=True)
os.makedirs(output_dir_gt, exist_ok=True)

# Iterate through each video file in the data
# for video_path, values in data.items():
#     # Extract video name from the filename
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     listener_fname = speaker_listener_dict[video_name][1]
#     # smooth out values['pred']
#     values['pred'] = smooth_logits_matrix(values['pred'])

#     # Iterate through frames
#     for frame_num, coefficients in enumerate(values['pred']):
#         # Extract head-pose and expression coefficients
#         head_pose = coefficients[:6]
#         expression = coefficients[6:]

#         # Create directories for video_id/frame_num
#         frame_dir = os.path.join(output_dir, f'{listener_fname}/{frame_num}')
#         os.makedirs(frame_dir, exist_ok=True)

#         # Save head-pose and expression as numpy arrays
#         np.save(os.path.join(frame_dir, 'pose.npy'), head_pose)
# #         np.save(os.path.join(frame_dir, 'exp.npy'), expression)

preds, gts, data_ids = data['y_pred'], data['y_true'], data['data_ids']
for i in range(len(preds)):
    pred = preds[i]
    gt = gts[i]
    data_id = data_ids[i].split('/')[-1].split('.')[0]
    pred = smooth_logits_matrix(pred)
    gt = smooth_logits_matrix(gt)
    
    for frame_num, coefficients in enumerate(pred):
        # Extract head-pose and expression coefficients
        head_pose = coefficients[:6]
        expression = coefficients[6:]

        # Create directories for video_id/frame_num
        frame_dir = os.path.join(output_dir_pred, f'{data_id}/{frame_num}')
        os.makedirs(frame_dir, exist_ok=True)

        # Save head-pose and expression as numpy arrays
        np.save(os.path.join(frame_dir, 'pose.npy'), head_pose)
        np.save(os.path.join(frame_dir, 'exp.npy'), expression)

    for frame_num, coefficients in enumerate(gt):
        # Extract head-pose and expression coefficients
        head_pose = coefficients[:6]
        expression = coefficients[6:]

        # Create directories for video_id/frame_num
        frame_dir = os.path.join(output_dir_gt, f'{data_id}/{frame_num}')
        os.makedirs(frame_dir, exist_ok=True)

        # Save head-pose and expression as numpy arrays
        np.save(os.path.join(frame_dir, 'pose.npy'), head_pose)
        np.save(os.path.join(frame_dir, 'exp.npy'), expression)