import os
import numpy as np
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm

REACT_TIME = 0.2 # 200ms or +/- 1 frame
data_path = '../data/databases/CANDOR/'
audio_feats_path = '../data/CANDOR_audio_feats/'
video_feats_path = '../data/emoca_sample_out1/'
processed_save_path = '../data/candor_processed/'

all_ids = os.listdir(data_path)
for file_id in tqdm(all_ids):
    try:
        cur_file_transcript = os.path.join(data_path, file_id, 'transcription', 'transcript_cliffhanger.csv')
        transcript_data = pd.read_csv(cur_file_transcript).values
        cur_file_audio = os.path.join(audio_feats_path, f'{file_id}.pkl')
        # audio_feats: 20ms (50 fps)
        with open(cur_file_audio, 'rb') as f:
            audio_feats = pickle.load(f)[0]
        # convert to 200ms (5 fps) by averaging 10 continuous frame
        audio_feats = np.array(audio_feats)
        audio_feats = audio_feats.reshape(-1, 10, 768)
        audio_feats = np.mean(audio_feats, axis=1)
        # find video file {file_id}_{speaker_id}.pkl
        all_speakers = np.unique(transcript_data[:, 1])
        # video_feats: 200ms (5 fps)
        video_feats = {}
        for speaker_id in all_speakers:
            cur_file_video = os.path.join(video_feats_path, f'{file_id}_{speaker_id}.pkl')
            with open(cur_file_video, 'rb') as f:
                video_feats[speaker_id] = pickle.load(f)
    except:
        print(f'Error processing {file_id}')
        continue
    # process transcript to extract utterance
    for row in transcript_data:
        try:
            utterance_id, speaker_id, start_time, end_time, _, _, _, _, _, _, _ = row
            start_time = float(start_time)
            end_time = float(end_time)
            # find video file
            cur_utt_speaker_feats = video_feats[speaker_id]
            # other speaker in all_speakers (only 2 speakers)
            listener_id = all_speakers[1] if speaker_id == all_speakers[0] else all_speakers[0]
            cur_utt_listener_feats = video_feats[listener_id]
            # find start and end index for speaker
            speaker_start_idx = int(start_time * 5)
            speaker_end_idx = int(end_time * 5)

            listener_start_idx = speaker_start_idx + 1
            listener_end_idx = speaker_end_idx + 1

            speaker_features = {}
            speaker_features['audio'] = audio_feats[speaker_start_idx:speaker_end_idx]
            speaker_features_video = []
            for idx in range(speaker_start_idx, speaker_end_idx):
                # convert i to 6 digits
                idx_str = str(idx).zfill(6)
                speaker_features_video.append(cur_utt_speaker_feats[idx_str])
            speaker_features['video'] = np.array(speaker_features_video)

            listener_features = {}
            listener_features_video = []
            for idx in range(listener_start_idx, listener_end_idx):
                # convert i to 6 digits
                idx_str = str(idx).zfill(6)
                listener_features_video.append(cur_utt_listener_feats[idx_str])
            listener_features['video'] = np.array(listener_features_video)

            # save to file
            unique_id = f'{file_id}_{utterance_id}'
            save_path_speaker = os.path.join(processed_save_path, 'speaker', f'{unique_id}.pkl')
            save_path_listener = os.path.join(processed_save_path, 'listener', f'{unique_id}.pkl')
            with open(save_path_speaker, 'wb') as f:
                pickle.dump(speaker_features, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(save_path_listener, 'wb') as f:
                pickle.dump(listener_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print(f'Error processing {file_id}_{utterance_id}')
            continue