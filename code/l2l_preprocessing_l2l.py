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
import torch
import pickle5 as pickle

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


for mode in ['train', 'test']:
    REACT_TIME = 0.2 # 200ms or +/- 1 frame
    data_path = f'../data/lm_listener_data/trevorconanstephen/segments_{mode}.pth'
    processed_save_path = '../data/l2l_l2l/'
    data = torch.load(data_path)
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

    output = []
    for i in range(len(data)):
        cur_item = data[i]
        p0_detail = cur_item['p0_detail'].numpy()
        p0_exp = cur_item['p0_exp'].numpy()
        p0_pose = cur_item['p0_pose'].numpy()

        p1_detail = cur_item['p1_detail'].numpy()
        p1_exp = cur_item['p1_exp'].numpy()
        p1_pose = cur_item['p1_pose'].numpy()

        target_size = 64
        num_bins = len(p0_detail)//target_size
        for j in range(num_bins):
            cur_p0_detail = p0_detail[j*target_size:(j+1)*target_size]
            cur_p0_exp = p0_exp[j*target_size:(j+1)*target_size]
            cur_p0_pose = p0_pose[j*target_size:(j+1)*target_size]

            cur_p1_detail = p1_detail[j*target_size:(j+1)*target_size]
            cur_p1_exp = p1_exp[j*target_size:(j+1)*target_size]
            cur_p1_pose = p1_pose[j*target_size:(j+1)*target_size]

            cur_p0 = np.concatenate([cur_p0_pose, cur_p0_exp], axis=-1)
            cur_p1 = np.concatenate([cur_p1_pose, cur_p1_exp], axis=-1)
            if len(cur_p0) == len(cur_p1) == target_size:
                cur_item_dict = {}
                cur_item_dict['video_speaker'] = cur_p1
                cur_item_dict['video_listener'] = cur_p0
                cur_item_dict['audio'] = np.zeros((256, 128))
                cur_item_dict['id'] = cur_item['fname']
                output.append(cur_item_dict)

    # np.save(os.path.join(processed_save_path, f'{mode}_elp.npy'), output)

            if len(cur_p0) == len(cur_p1) == 64:
                p0_list_faces_clean_deca.append(cur_p0)
                p1_list_faces_clean_deca.append(cur_p0)
                p0_speak_audio_clean_deca.append(np.zeros((256, 128)))
                p1_speak_audio_clean_deca.append(np.zeros((256, 128)))
                p0_speak_faces_clean_deca.append(cur_p1)
                p1_speak_faces_clean_deca.append(cur_p1)
                tmp = []
                for k in range(target_size):
                    tmp.append([cur_item['fname'], '0', str(int(cur_item['start']*25) + k + target_size*j)])
                p1_speak_files_clean_deca.append(tmp)
                p0_speak_files_clean_deca.append(tmp)

    p0_list_faces_clean_deca = np.array(p0_list_faces_clean_deca)
    p0_speak_audio_clean_deca = np.array(p0_speak_audio_clean_deca)
    p0_speak_faces_clean_deca = np.array(p0_speak_faces_clean_deca)
    p0_speak_files_clean_deca = np.array(p0_speak_files_clean_deca)
    p1_list_faces_clean_deca = np.array(p1_list_faces_clean_deca)
    p1_speak_audio_clean_deca = np.array(p1_speak_audio_clean_deca)
    p1_speak_faces_clean_deca = np.array(p1_speak_faces_clean_deca)
    p1_speak_files_clean_deca = np.array(p1_speak_files_clean_deca)

    print(len(p0_list_faces_clean_deca))
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
    