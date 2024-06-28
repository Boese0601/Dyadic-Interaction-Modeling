import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 
from s3prl.nn import S3PRLUpstream
import pickle5 as pickle

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train",read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio

    # def downsample_mean(self, array, new_t):
    #     t, d = array.shape
    #     # new_t = int(t * factor)
    #     downsampled = np.zeros((new_t, d))
        
    #     window_size = int(t / new_t)
        
    #     for i in range(new_t):
    #         start = i * window_size
    #         end = start + window_size
    #         downsampled[i] = np.mean(array[start:end], axis=0)
        
    #     return downsampled
        
    def downsample_mean(self, array, new_t):
        torch_array = torch.from_numpy(array).unsqueeze(0)
        _,t, dim = torch_array.shape
        torch_array = torch_array.permute(0,2,1)
        downsampled = torch.nn.functional.interpolate(torch_array,size=(new_t),mode='linear', align_corners=True)
        downsampled = downsampled.permute(0,2,1).squeeze(0).numpy()
        return downsampled

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        audio = self.downsample_mean(audio, vertice.shape[0]) 
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            # one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            emoca_feats = self.data[index]["emoca"]
        else:
            # one_hot = self.one_hot_labels
            emoca_feats = self.data[index]["emoca"]
        # emoca_feats[:, 0:6] = 0
        if self.read_audio:
            return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(emoca_feats), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(emoca_feats), file_name

    def __len__(self):
        return self.len

def read_data():
    DATA_ROOT = '../data/BIWI_data/'
    emoca_biwi_root = os.path.join(DATA_ROOT, 'emoca_biwi')
    wav_path = 'wav'
    vertices_path = 'vertices_npy'
    template_file = 'templates.pkl'
    read_audio = True
    dataset = 'BIWI'
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    model = S3PRLUpstream("hubert")
    pretrained_ckpt = '../data/s3prl_clean/s3prl/s3prl/result/downstream/iemocap_dan/dev-best.ckpt'
    state_dict = torch.load(pretrained_ckpt)['Upstream']
    modified_state_dict = {}
    for k, v in state_dict.items():
        modified_state_dict['upstream.'+k] = v
    model.cuda()
    model.eval()

    audio_path = os.path.join(DATA_ROOT, wav_path)
    vertices_path = os.path.join(DATA_ROOT, vertices_path)
    if read_audio: # read_audio==False when training vq to save time
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(DATA_ROOT, template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                try:
                    if read_audio:
                        wav_path = os.path.join(r,f)
                        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                        # input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                        with torch.no_grad():
                            wavs = torch.FloatTensor(speech_array).unsqueeze(0).cuda()
                            wavs_len = torch.LongTensor([wavs.shape[1]]).cuda()
                            all_hs, all_hs_len = model(wavs, wavs_len)
                            input_values = all_hs[-1].squeeze().cpu().detach().numpy()
                    key = f.replace("wav", "npy")
                    data[key]["audio"] = input_values if read_audio else None
                    subject_id = "_".join(key.split("_")[:-1])
                    temp = templates[subject_id]
                    data[key]["name"] = f
                    data[key]["template"] = temp.reshape((-1)) 
                    vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                    emoca_biwi_path = os.path.join(emoca_biwi_root, f.split(".")[0]+'.pkl')
                    if not os.path.exists(vertice_path):
                        del data[key]
                    else:
                        if dataset == "vocaset":
                            data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                        elif dataset == "BIWI":
                            data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
                    # load emoca
                    with open(emoca_biwi_path, 'rb') as fin:
                        emoca_data = pickle.load(fin)
                    emoca_data_list = []
                    sorted_frames = sorted(list(emoca_data.keys()))
                    for frame in sorted_frames:
                        cur_frame_exp = emoca_data[frame]['exp']
                        cur_frame_pose = emoca_data[frame]['pose']
                        cur_frame_data = np.concatenate([cur_frame_pose, cur_frame_exp])
                        emoca_data_list.append(cur_frame_data)
                    data[key]["emoca"] = np.array(emoca_data_list)
                except:
                    continue
    train_subjects = "F2 F3 F4 M3 M4 M5"
    val_subjects = "F2 F3 F4 M3 M4 M5"
    test_subjects = "F1 F5 F6 F7 F8 M1 M2 M6"
    subjects_dict = {}
    subjects_dict["train"] = [i for i in train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in test_subjects.split(" ")]


    #train vq and pred
    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
    'BIWI':{'train':range(1,33),'val':range(37,41),'test':range(37,41)}}


    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[dataset]['test']:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(batch_size):
    read_audio = True
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data()
    train_data = Dataset(train_data,subjects_dict,"train",read_audio)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_data = Dataset(valid_data,subjects_dict,"val",read_audio)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=4)
    test_data = Dataset(test_data,subjects_dict,"test",read_audio)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)
    return dataset

def get_dataloaders_convert(batch_size):
    read_audio = False
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data()
    train_data = Dataset(train_data,subjects_dict,"train",read_audio)
    valid_data = Dataset(valid_data,subjects_dict,"val",read_audio)
    test_data = Dataset(test_data,subjects_dict,"test",read_audio)
    data_all = torch.utils.data.ConcatDataset([train_data, valid_data, test_data])
    dataset["train"] = data.DataLoader(dataset=data_all, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset

if __name__ == "__main__":
    dataset = get_dataloaders(32)
