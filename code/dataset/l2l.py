import torch 
import numpy as np
import pdb
import argparse
from scipy.io import wavfile
import os
import math
from scipy.io import wavfile
import librosa
from PIL import Image
import random
import torch
from s3prl.nn import S3PRLUpstream
import os
import torchaudio
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle5 as pickle
import sys
from torch.utils import data 

def downsample_mean(array, new_t):
    torch_array = torch.from_numpy(array).unsqueeze(0)
    _,t, _ = torch_array.shape
    torch_array = torch_array.permute(0,2,1)
    downsampled = torch.nn.functional.interpolate(torch_array,size=(new_t),mode='linear', align_corners=True)
    downsampled = downsampled.permute(0,2,1).squeeze(0).numpy()
    return downsampled

class LmListenerDataset(data.Dataset):
    def __init__(self, data_path, mode='train'):
        self.abs_path = os.path.join(data_path, f'segments_{mode}.pth')
        cur_data = torch.load(self.abs_path)
        self.data = []
        self.cur_data = cur_data
        for i in range(len(cur_data)):
            if 'hubert_feat' not in cur_data[i].keys():
                continue
            start_time, end_time = cur_data[i]['split_start_time'], cur_data[i]['split_end_time']
            if start_time == end_time:
                continue
            if len(cur_data[i]['p0_exp']) == len(cur_data[i]['p1_exp']) and len(cur_data[i]['p0_exp']) >= 24:
                cur_data[i]['hubert_feat'] = downsample_mean(cur_data[i]['hubert_feat'],cur_data[i]['p0_exp'].shape[0])
                if len(cur_data[i]['p0_exp']) < 1024:
                    self.data.append(cur_data[i])
                else:
                    # break into chunks of 1024
                    num_chunks = len(cur_data[i]['p0_exp']) // 1024
                    for j in range(num_chunks):
                        new_item = {}
                        new_item['p0_exp'] = cur_data[i]['p0_exp'][j*1024:(j+1)*1024]
                        new_item['p1_exp'] = cur_data[i]['p1_exp'][j*1024:(j+1)*1024]
                        new_item['p0_pose'] = cur_data[i]['p0_pose'][j*1024:(j+1)*1024]
                        new_item['p1_pose'] = cur_data[i]['p1_pose'][j*1024:(j+1)*1024]
                        new_item['hubert_feat'] = cur_data[i]['hubert_feat'][j*1024:(j+1)*1024]
                        # new_item['fname'] = cur_data[i]['fname'] + '#' +  str(start_time) + '#' + str(end_time)
                        new_item['fname'] = str(i) + '**' + str(start_time) + '**' + str(end_time) + '**' + cur_data[i]['fname']
                        self.data.append(new_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cur_item = self.data[index]
        cur_speaker_exp = torch.FloatTensor(cur_item['p1_exp'])
        cur_speaker_pose = torch.FloatTensor(cur_item['p1_pose'])
        cur_listener_exp = torch.FloatTensor(cur_item['p0_exp'])
        cur_listener_pose = torch.FloatTensor(cur_item['p0_pose'])
        cur_filename = cur_item['fname']
        cur_speaker_feats = torch.cat((cur_speaker_pose, cur_speaker_exp), dim=1)
        cur_listener_feats = torch.cat((cur_listener_pose, cur_listener_exp), dim=1)
        audio_feats = torch.FloatTensor(cur_item['hubert_feat'])
        combined_feats = torch.cat((cur_speaker_feats, audio_feats), dim=1)
        output = (combined_feats, cur_listener_feats, cur_filename)
        return output

def pad_collate_lm(batch): 
    (xx, yy, zz) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    zz_names = [z for z in zz]
    return xx_pad, yy_pad, x_lens, y_lens, zz_names

def get_lm_listener_dataloaders(batch_size):
    dataset = {}
    train_data = LmListenerDataset(
        data_path='../data/lm_listener_data/hubert_dict/', 
        mode='train'
    )
    val_data = LmListenerDataset(
        data_path='../data/lm_listener_data/hubert_dict/', 
        mode='test'
    )
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate_lm)
    dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=pad_collate_lm)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    lm_dataset = get_lm_listener_dataloaders(args.batch_size)
    for batch_data in lm_dataset["train"]:
        (xx_pad, yy_pad, x_lens, y_lens, zz_names) = batch_data
    for batch_data in lm_dataset["valid"]:
        (xx_pad, yy_pad, x_lens, y_lens, zz_names) = batch_data
