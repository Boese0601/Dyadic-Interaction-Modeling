import os
import numpy as np
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import glob
import librosa
from PIL import Image
import random
import pandas as pd

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

def load_melspec(audio_path, num_frames):
    waveform, sample_rate = librosa.load('{}'.format(audio_path), sr=16000)
    win_len = int(0.025*sample_rate)
    hop_len = int(0.010*sample_rate)
    fft_len = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
    S_dB = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, hop_length=hop_len, win_length=win_len, n_fft=fft_len)

    ## do some resizing to match frame rate
    im = Image.fromarray(S_dB)
    _, feature_dim = im.size
    scale_four = num_frames*4
    im = im.resize((scale_four, feature_dim), Image.LANCZOS)
    S_dB = np.array(im)
    return S_dB

mode = 'test'
REACT_TIME = 0.2 # 200ms or +/- 1 frame
data_path = '../data/vico_dataset/'
audio_feats_path = '../data/vico_dataset/hubert/'
video_feats_path = '../data/vico_dataset/emoca/'
processed_save_path = '../data/vico_l2l/'

vico_original_root = '../data/vico_dataset/audios/'

metadata_filepath = '../data/RLD_data.csv'
metadata = pd.read_csv(metadata_filepath).values
train_list, test_list = [], []
for i in range(metadata.shape[0]):
    if metadata[i, -1] == 'train':
        train_list.append(metadata[i, 1])
    elif metadata[i, -1] == 'test':
        test_list.append(metadata[i, 1])
# |-- p0_list_faces_clean_deca.npy (8763, 64, 184)
# |-- p0_speak_audio_clean_deca.npy (8091, 256, 128)
# |-- p0_speak_faces_clean_deca.npy (8091, 64, 184)
# |-- p0_speak_files_clean_deca.npy (8091, 64, 3)
# |-- p1_list_faces_clean_deca.npy (8091, 64, 184)
# |-- p1_speak_audio_clean_deca.npy (8763, 256, 128)
# |-- p1_speak_faces_clean_deca.npy (8763, 64, 184)
# |-- p1_speak_files_clean_deca.npy (8763, 64, 3)

p0_list_faces_clean_deca = []
p0_speak_audio_clean_deca = []
p0_speak_faces_clean_deca = []
p0_speak_files_clean_deca = []
p1_list_faces_clean_deca = []
p1_speak_audio_clean_deca = []
p1_speak_faces_clean_deca = []
p1_speak_files_clean_deca = []
# .0: listener, .1: speaker
all_ids = os.listdir(audio_feats_path)

metadata_path = '../data/RLD_data.csv'

metadata = pd.read_csv(metadata_path).values
speaker_listener_dict = {}
for row in metadata:
    file_id = row[1]
    listener_file = row[2]
    speaker_file = row[3]
    speaker_listener_dict[file_id] = (speaker_file, listener_file)

xx = 0
for file_id in tqdm(all_ids):
    unique_id = file_id.split('.')[0]
    if mode == 'train':
        if unique_id not in train_list:
            continue
    elif mode == 'test':
        if unique_id not in test_list:
            continue

    try:
        cur_speaker_file, cur_listener_file = speaker_listener_dict[unique_id]
    except:
        print(f'erron on {unique_id}')
        continue

    # cur_file_audio = os.path.join(audio_feats_path, f'{unique_id}.pkl')
    # # audio_feats: 20ms (50 fps)
    # with open(cur_file_audio, 'rb') as f:
    #     audio_feats = pickle.load(f)[0]
    # # convert to 10 fps
    # audio_feats = np.array(audio_feats)

    # truncate length so that it's divisible by 5
    # audio_feats = audio_feats[:-(audio_feats.shape[0] % 5)]
    # audio_feats = audio_feats.reshape(-1, 5, 768)
    # audio_feats = np.mean(audio_feats, axis=1)

    # audio_feats = downsample_mean(audio_feats, factor=0.6) # 50 fps -> 30 fps

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
            cur_frame_data_detail_path = os.path.join(cur_frame_data_path, 'detail.npy')

            cur_exp_data = np.load(cur_frame_data_exp_path)
            cur_pose_data = np.load(cur_frame_data_pose_path)
            cur_detail_data = np.load(cur_frame_data_detail_path)
            cur_frame_data = np.concatenate([cur_exp_data, cur_pose_data, cur_detail_data], axis=0)
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
            cur_frame_data_detail_path = os.path.join(cur_frame_data_path, 'detail.npy')

            cur_exp_data = np.load(cur_frame_data_exp_path)
            cur_pose_data = np.load(cur_frame_data_pose_path)
            cur_detail_data = np.load(cur_frame_data_detail_path)
            cur_frame_data = np.concatenate([cur_exp_data, cur_pose_data, cur_detail_data], axis=0)
            video_feats_listener.append(cur_frame_data)
            idx += 1
    video_feats_listener = np.array(video_feats_listener)

    cur_file_audio = os.path.join(vico_original_root, f'{unique_id}.wav')
    mfcc_feats = load_melspec(cur_file_audio, min(video_feats.shape[0], video_feats_listener.shape[0]))
    mfcc_feats = mfcc_feats.transpose(1, 0)

    coin_flip = random.random()
    xx += 1
    if coin_flip < 1.0:
        # add to p0

        # segment video features into 64 continuous frames
        num_frames = 64
        num_segments = video_feats.shape[0] // num_frames
        for i in range(num_segments-1): # we don't want the last segment because it's not 64 frames
            cur_video_feats = video_feats[i*num_frames:(i+1)*num_frames]
            cur_video_feats_listener = video_feats_listener[i*num_frames:(i+1)*num_frames]
            cur_mfcc_feats = mfcc_feats[i*num_frames*4:(i+1)*num_frames*4]
            if len(cur_video_feats) == len(cur_video_feats_listener) == 64 and len(cur_mfcc_feats) == 64*4:
                p1_list_faces_clean_deca.append(cur_video_feats_listener)
                p0_speak_audio_clean_deca.append(cur_mfcc_feats)
                p0_speak_faces_clean_deca.append(cur_video_feats)
                tmp = []
                for j in range(num_frames):
                    tmp.append([file_id, '0', str(i*num_frames+j)])
                p0_speak_files_clean_deca.append(tmp)
    else:
        # add to p1

        # segment video features into 64 continuous frames
        num_frames = 64
        num_segments = video_feats.shape[0] // num_frames
        for i in range(num_segments-1):
            cur_video_feats = video_feats[i*num_frames:(i+1)*num_frames]
            cur_video_feats_listener = video_feats_listener[i*num_frames:(i+1)*num_frames]
            cur_mfcc_feats = mfcc_feats[i*num_frames*4:(i+1)*num_frames*4]
            if len(cur_video_feats) == len(cur_video_feats_listener) == 64 and len(cur_mfcc_feats) == 64*4:
                p0_list_faces_clean_deca.append(cur_video_feats_listener)
                p1_speak_audio_clean_deca.append(cur_mfcc_feats)
                p1_speak_faces_clean_deca.append(cur_video_feats)
                tmp = []
                for j in range(num_frames):
                    tmp.append([file_id, '1', str(i*num_frames+j)])
                p1_speak_files_clean_deca.append(tmp)

p0_list_faces_clean_deca = np.array(p0_list_faces_clean_deca)
p0_speak_audio_clean_deca = np.array(p0_speak_audio_clean_deca)
p0_speak_faces_clean_deca = np.array(p0_speak_faces_clean_deca)
p0_speak_files_clean_deca = np.array(p0_speak_files_clean_deca)
p1_list_faces_clean_deca = np.array(p1_list_faces_clean_deca)
p1_speak_audio_clean_deca = np.array(p1_speak_audio_clean_deca)
p1_speak_faces_clean_deca = np.array(p1_speak_faces_clean_deca)
p1_speak_files_clean_deca = np.array(p1_speak_files_clean_deca)

# save output path 
save_output_path = os.path.join(processed_save_path, mode)
os.makedirs(save_output_path, exist_ok=True)
np.save(os.path.join(save_output_path, 'p0_list_faces_clean_deca.npy'), p0_list_faces_clean_deca)
np.save(os.path.join(save_output_path, 'p0_speak_audio_clean_deca.npy'), p0_speak_audio_clean_deca)
np.save(os.path.join(save_output_path, 'p0_speak_faces_clean_deca.npy'), p0_speak_faces_clean_deca)
np.save(os.path.join(save_output_path, 'p0_speak_files_clean_deca.npy'), p0_speak_files_clean_deca)
np.save(os.path.join(save_output_path, 'p1_list_faces_clean_deca.npy'), p1_list_faces_clean_deca)
np.save(os.path.join(save_output_path, 'p1_speak_audio_clean_deca.npy'), p1_speak_audio_clean_deca)
np.save(os.path.join(save_output_path, 'p1_speak_faces_clean_deca.npy'), p1_speak_faces_clean_deca)
np.save(os.path.join(save_output_path, 'p1_speak_files_clean_deca.npy'), p1_speak_files_clean_deca)

print('here')
    