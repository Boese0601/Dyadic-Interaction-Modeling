import os
import numpy as np
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm

def downsample_mean(array, factor=0.6):
    t, d = array.shape
    new_t = int(t * factor)
    downsampled = np.zeros((new_t, d))
    
    window_size = int(t / new_t)
    
    for i in range(new_t):
        start = i * window_size
        end = start + window_size
        downsampled[i] = np.mean(array[start:end], axis=0)
    
    return downsampled



REACT_TIME = 0.2 # 200ms or +/- 1 frame
data_path = '../data/vico_dataset/'
audio_feats_path = '../data/vico_dataset/hubert/'
video_feats_path = '../data/vico_dataset/emoca/'
processed_save_path = '../data/vico_processed_30fps/'
metadata_path = '../data/RLD_data.csv'

metadata = pd.read_csv(metadata_path).values
speaker_listener_dict = {}
for row in metadata:
    file_id = row[1]
    listener_file = row[2]
    speaker_file = row[3]
    speaker_listener_dict[file_id] = (speaker_file, listener_file)

all_ids = os.listdir(audio_feats_path)
for file_id in tqdm(all_ids):
    unique_id = file_id.split('.')[0]
    cur_file_audio = os.path.join(audio_feats_path, f'{unique_id}.pkl')
    # audio_feats: 20ms (50 fps)
    with open(cur_file_audio, 'rb') as f:
        audio_feats = pickle.load(f)[0]
    # convert to 10 fps
    audio_feats = np.array(audio_feats)
    try:
        cur_speaker_file, cur_listener_file = speaker_listener_dict[unique_id]
    except:
        print(f'erron on {unique_id}')
        continue

    # truncate length so that it's divisible by 5
    # audio_feats = audio_feats[:-(audio_feats.shape[0] % 5)]
    # audio_feats = audio_feats.reshape(-1, 5, 768)
    # audio_feats = np.mean(audio_feats, axis=1)

    audio_feats = downsample_mean(audio_feats, factor=0.6) # 50 fps -> 30 fps

    # video speaker path
    cur_file_video_speaker = os.path.join(video_feats_path, cur_speaker_file, 'EMOCA_v2_lr_mse_20')
    all_frame_data = list(sorted(os.listdir(cur_file_video_speaker)))
    video_feats = []
    idx = 0
    for frame_data in all_frame_data:
        if frame_data.startswith('0'):
            cur_frame_data_path = os.path.join(cur_file_video_speaker, frame_data)
            # if idx % 3 == 0:
            cur_frame_data_exp_path = os.path.join(cur_frame_data_path, 'exp.npy')
            cur_frame_data_pose_path = os.path.join(cur_frame_data_path, 'pose.npy')

            cur_exp_data = np.load(cur_frame_data_exp_path)
            cur_pose_data = np.load(cur_frame_data_pose_path)
            cur_frame_data = np.concatenate([cur_pose_data, cur_exp_data], axis=0)
            video_feats.append(cur_frame_data)

            idx += 1
    video_feats = np.array(video_feats)

    # video listener path
    cur_file_video_listener = os.path.join(video_feats_path, cur_listener_file, 'EMOCA_v2_lr_mse_20')
    all_frame_data = list(sorted(os.listdir(cur_file_video_listener)))
    video_feats_listener = []
    idx = 0
    for frame_data in all_frame_data:
        if frame_data.startswith('0'):
            cur_frame_data_path = os.path.join(cur_file_video_listener, frame_data)
            # if idx % 3 == 0:
            cur_frame_data_exp_path = os.path.join(cur_frame_data_path, 'exp.npy')
            cur_frame_data_pose_path = os.path.join(cur_frame_data_path, 'pose.npy')

            cur_exp_data = np.load(cur_frame_data_exp_path)
            cur_pose_data = np.load(cur_frame_data_pose_path)
            cur_frame_data = np.concatenate([cur_pose_data, cur_exp_data], axis=0)
            video_feats_listener.append(cur_frame_data)
            idx += 1
    video_feats_listener = np.array(video_feats_listener)

    # make sure audio, video_speaker, video_listener have the same length
    # print(audio_feats.shape, video_feats.shape, video_feats_listener.shape)
    min_len = min(audio_feats.shape[0], video_feats.shape[0], video_feats_listener.shape[0])
    audio_feats = audio_feats[:min_len]
    video_feats = video_feats[:min_len]
    video_feats_listener = video_feats_listener[:min_len]

    output_dict = {}
    output_dict['audio'] = audio_feats
    output_dict['video_speaker'] = video_feats
    output_dict['video_listener'] = video_feats_listener

    output_file_name = f'{unique_id}.pkl'
    output_file_path = os.path.join(processed_save_path, output_file_name)

    # save output_dict to file
    with open(output_file_path, 'wb') as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)