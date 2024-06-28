import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 
import random
import pickle as pickle
import pandas as pd

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train",read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        if self.read_audio:
            return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len
    
class CandorListenerDataset(data.Dataset):
    def __init__(self, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        video_feats = data['video']
        return torch.FloatTensor(video_feats), file_name
    
    def __len__(self):
        return self.len
    
class CandorSpeakerDataset(data.Dataset):
    def __init__(self, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        video_feats = torch.FloatTensor(data['video'])
        audio_feats = torch.FloatTensor(data['audio'])
        # print(video_feats.shape, audio_feats.shape)
        # combined_feats = torch.cat((video_feats, audio_feats), dim=1)
        return video_feats, file_name
    
    def __len__(self):
        return self.len
    
class CandorDataset(data.Dataset):
    def __init__(self, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name_speaker, file_name_listener = self.data[index]
        with open(file_name_speaker, 'rb') as f:
            data_speaker = pickle.load(f)
        with open(file_name_listener, 'rb') as f:
            data_listener = pickle.load(f)
        video_feats_speaker = torch.FloatTensor(data_speaker['video'])
        audio_feats_speaker = torch.FloatTensor(data_speaker['audio'])
        video_feats_listener = torch.FloatTensor(data_listener['video'])
        combined_feats = torch.cat((video_feats_speaker, audio_feats_speaker), dim=1)
        # (combined_feats, video_feats_listener, self.data[index], speaker_id, listener_id)
        output = (combined_feats, video_feats_listener, None, 0, 0)
        return output
    
    def __len__(self):
        return self.len
    
class ViCoDataset(data.Dataset):
    def __init__(self, data_path, meta_data_path, mode='train'):
        meta_data = pd.read_csv(meta_data_path).values
        self.data_ids = []
        for i in range(len(meta_data)):
            if meta_data[i, 6] == mode:
                self.data_ids.append(meta_data[i, 1])
        self.data = []
        for i in range(len(self.data_ids)):
            if os.path.exists(os.path.join(data_path, self.data_ids[i]+'.pkl')):
                cur_data = pickle.load(open(os.path.join(data_path, self.data_ids[i]+'.pkl'), 'rb'))
                if len(cur_data['video_speaker']) == len(cur_data['audio']) == len(cur_data['video_listener']) and 1024 >= len(cur_data['video_speaker']) >= 5:
                    self.data.append(os.path.join(data_path, self.data_ids[i]+'.pkl'))
        print(f'Loaded {len(self.data)} data points for {mode}')

        self.id2speaker_id, self.id2listener_id = {}, {}
        self.id2sentiment = {}
        sentiment2idx = {
            'neutral': 0,
            'positive': 1,
            'negative': 2
        }
        for i in range(len(meta_data)):
            self.id2speaker_id[meta_data[i, 1]] = meta_data[i, 5]
            self.id2listener_id[meta_data[i, 1]] = meta_data[i, 4]
            self.id2sentiment[meta_data[i, 1]] = sentiment2idx[meta_data[i, 0]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as f:
            data = pickle.load(f)
        unique_id = self.data[index].split('/')[-1].split('.')[0]
        speaker_id = self.id2speaker_id[unique_id]
        listener_id = self.id2listener_id[unique_id]
        cur_sentiment = self.id2sentiment[unique_id]

        video_feats_speaker = torch.FloatTensor(data['video_speaker'])
        video_feats_speaker = torch.ones_like(video_feats_speaker)
        video_feats_listener = torch.FloatTensor(data['video_listener'])
        audio_feats = torch.FloatTensor(data['audio'])
        combined_feats = torch.cat((video_feats_speaker, audio_feats), dim=1)
        output = (combined_feats, video_feats_listener, self.data[index], speaker_id, listener_id, cur_sentiment)
        return output
    
class ViCoSpeakerDataset(data.Dataset):
    def __init__(self, data_path, meta_data_path, mode='train'):
        meta_data = pd.read_csv(meta_data_path).values
        self.data_ids = []
        for i in range(len(meta_data)):
            if meta_data[i, 6] == mode:
                self.data_ids.append(meta_data[i, 1])
        self.data = []
        for i in range(len(self.data_ids)):
            if os.path.exists(os.path.join(data_path, self.data_ids[i]+'.pkl')):
                cur_data = pickle.load(open(os.path.join(data_path, self.data_ids[i]+'.pkl'), 'rb'))
                if len(cur_data['video_speaker']) == len(cur_data['audio']) == len(cur_data['video_listener']) and 1024 >= len(cur_data['video_speaker']) >= 5:
                    self.data.append(os.path.join(data_path, self.data_ids[i]+'.pkl'))
        print(f'Loaded {len(self.data)} data points for {mode}')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as f:
            data = pickle.load(f)
        video_feats_speaker = torch.FloatTensor(data['video_speaker'])
        video_feats_listener = torch.FloatTensor(data['video_listener'])
        # audio_feats = torch.FloatTensor(data['audio'])
        # combined_feats = torch.cat((video_feats_speaker, audio_feats), dim=1)
        return video_feats_speaker, self.data[index]
    
class ViCoListenerDataset(data.Dataset):
    def __init__(self, data_path, meta_data_path, mode='train'):
        meta_data = pd.read_csv(meta_data_path).values
        self.data_ids = []
        for i in range(len(meta_data)):
            if meta_data[i, 6] == mode:
                self.data_ids.append(meta_data[i, 1])
        self.data = []
        for i in range(len(self.data_ids)):
            if os.path.exists(os.path.join(data_path, self.data_ids[i]+'.pkl')):
                cur_data = pickle.load(open(os.path.join(data_path, self.data_ids[i]+'.pkl'), 'rb'))
                if len(cur_data['video_speaker']) == len(cur_data['audio']) == len(cur_data['video_listener']) and 1024 >= len(cur_data['video_speaker']) >= 5:
                    self.data.append(os.path.join(data_path, self.data_ids[i]+'.pkl'))
        print(f'Loaded {len(self.data)} data points for {mode}')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as f:
            data = pickle.load(f)
        video_feats_speaker = torch.FloatTensor(data['video_speaker'])
        video_feats_listener = torch.FloatTensor(data['video_listener'])
        audio_feats = torch.FloatTensor(data['audio'])
        # print(video_feats_listener.sum())
        return video_feats_listener, self.data[index]
    
class LmListenerDataset(data.Dataset):
    def __init__(self, data_path, mode='train'):
        self.abs_path = os.path.join(data_path, f'segments_{mode}.pth')
        cur_data = torch.load(self.abs_path)
        self.data = []
        for i in range(len(cur_data)):
            if len(cur_data[i]['p0_exp']) == len(cur_data[i]['p1_exp']) and len(cur_data[i]['p0_exp']) >= 24:
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
                        new_item['fname'] = cur_data[i]['fname']
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

        audio_feats = torch.zeros((cur_speaker_exp.shape[0], 768))
        combined_feats = torch.cat((cur_speaker_feats, audio_feats), dim=1)
        output = (combined_feats, cur_listener_feats, cur_filename)
        return output

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_root, args.wav_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)
    if args.read_audio: # read_audio==False when training vq to save time
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)

    template_file = os.path.join(args.data_root, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                if args.read_audio:
                    wav_path = os.path.join(r,f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values if args.read_audio else None
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]


    #train vq and pred
    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
    'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}


    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict

def read_data_candor_listener():
    listener_data_root = '../data/candor_processed/listener'
    all_data = os.listdir(listener_data_root)
    unique_ids = list(set([i.split('_')[0] for i in all_data]))
    # train data: 95%, val data: 5%
    random.shuffle(unique_ids)
    train_ids = unique_ids[:int(len(unique_ids)*0.95)]
    val_ids = unique_ids[int(len(unique_ids)*0.95):]

    train_data = []
    val_data = []
    for file_id in all_data:
        cur_path = os.path.join(listener_data_root, file_id)
        with open(cur_path, 'rb') as f:
            cur_data = pickle.load(f)
        if not 5 <= len(cur_data['video']) <= 150:
            continue
        if file_id.split('_')[0] in train_ids:
            train_data.append(os.path.join(listener_data_root, file_id))
        else:
            val_data.append(os.path.join(listener_data_root, file_id))
    return train_data, val_data

def read_data_candor_speaker():
    speaker_data_root = '../data/candor_processed/speaker'
    all_data = os.listdir(speaker_data_root)
    unique_ids = list(set([i.split('_')[0] for i in all_data]))
    # train data: 95%, val data: 5%
    random.shuffle(unique_ids)
    train_ids = unique_ids[:int(len(unique_ids)*0.99)]
    val_ids = unique_ids[int(len(unique_ids)*0.99):]

    train_data = []
    val_data = []
    for file_id in all_data:
        cur_path = os.path.join(speaker_data_root, file_id)
        with open(cur_path, 'rb') as f:
            cur_data = pickle.load(f)
        # if len(cur_data['audio']) != len(cur_data['video']):
        #     continue
        if not 25 <= len(cur_data['video']) <= 150:
            continue
        if file_id.split('_')[0] in train_ids:
            train_data.append(os.path.join(speaker_data_root, file_id))
        else:
            val_data.append(os.path.join(speaker_data_root, file_id))
    return train_data, val_data

def read_data_candor():
    speaker_data_root = '../data/candor_processed/speaker'
    listener_data_root = '../data/candor_processed/listener'
    all_data = os.listdir(speaker_data_root)
    unique_ids = list(set([i.split('_')[0] for i in all_data]))
    # set seed for shuffling
    random.seed(42)
    random.shuffle(unique_ids)
    train_ids = unique_ids[:int(len(unique_ids)*0.95)]
    train_data = []
    val_data = []
    for file_id in all_data:
        cur_path_speaker = os.path.join(speaker_data_root, file_id)
        cur_path_listener = os.path.join(listener_data_root, file_id)

        if not os.path.exists(cur_path_listener):
            continue

        with open(cur_path_speaker, 'rb') as f:
            cur_data_speaker = pickle.load(f)
        with open(cur_path_listener, 'rb') as f:
            cur_data_listener = pickle.load(f)
        if not (5 <= len(cur_data_speaker['video']) <= 250) or len(cur_data_speaker['audio']) != len(cur_data_speaker['video']):
            continue
        if len(cur_data_speaker['video']) != len(cur_data_listener['video']):
            continue
        if file_id.split('_')[0] in train_ids:
            train_data.append((cur_path_speaker, cur_path_listener))
        else:
            val_data.append((cur_path_speaker, cur_path_listener))
    return train_data, val_data


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train",args.read_audio)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_data = Dataset(valid_data,subjects_dict,"val",args.read_audio)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.workers)
    test_data = Dataset(test_data,subjects_dict,"test",args.read_audio)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=args.workers)
    return dataset

def get_candor_listener_dataloaders(args):
    dataset = {}
    train_data, val_data = read_data_candor_listener()
    train_data = CandorListenerDataset(train_data)
    val_data = CandorListenerDataset(val_data)
    if not args.distributed:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(train_data))
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(val_data))
    return dataset

def get_candor_speaker_dataloaders(args):
    dataset = {}
    train_data, val_data = read_data_candor_speaker()
    train_data = CandorSpeakerDataset(train_data)
    val_data = CandorSpeakerDataset(val_data)
    if not args.distributed:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(train_data))
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(val_data))
    return dataset



def pad_collate(batch):
    (xx, yy, zz, speaker_ids, listener_ids, sentiment) = zip(*batch)
    x_lens = [len(x) for x in xx]
    # y_lens = [len(y) for y in yy]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    zz_names = [z for z in zz]
    speaker_ids = torch.LongTensor(speaker_ids)
    listener_ids = torch.LongTensor(listener_ids)
    sentiment = torch.LongTensor(sentiment)
    return xx_pad, yy_pad, x_lens, (speaker_ids, listener_ids), zz_names

def pad_collate_lm(batch):
    (xx, yy, zz) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    zz_names = [z for z in zz]
    return xx_pad, yy_pad, x_lens, y_lens, zz_names

def get_candor_dataloaders(batch_size):
    dataset = {}
    train_data, val_data = read_data_candor()
    train_data = CandorDataset(train_data)
    val_data = CandorDataset(val_data)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate)
    dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate)
    # dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=4, sampler=data.distributed.DistributedSampler(train_data), collate_fn=pad_collate)
    # dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4, sampler=data.distributed.DistributedSampler(val_data), collate_fn=pad_collate)
    return dataset

def get_vico_dataloaders(batch_size):
    dataset = {}
    train_data = ViCoDataset(
        data_path='../data/vico_processed_30fps', 
        meta_data_path='../data/RLD_data.csv', 
        mode='train'
    )
    val_data = ViCoDataset(
        data_path='../data/vico_processed_30fps', 
        meta_data_path='../data/RLD_data.csv', 
        mode='test'
    )
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate)
    dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate)
    #combine train_data and val_data
    dataset_all = data.ConcatDataset([train_data, val_data])
    dataset["all"] = data.DataLoader(dataset=dataset_all, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate)
    return dataset

def get_vico_speaker_dataloaders(args):
    dataset = {}
    train_data = ViCoSpeakerDataset(
        data_path='../data/vico_processed_30fps', 
        meta_data_path='../data/RLD_data.csv', 
        mode='train'
    )
    val_data = ViCoSpeakerDataset(
        data_path='../data/vico_processed_30fps', 
        meta_data_path='../data/RLD_data.csv', 
        mode='test'
    )
    if not args.distributed:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(train_data))
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(val_data))
    return dataset

def get_vico_listener_dataloaders(args):
    dataset = {}
    train_data = ViCoListenerDataset(
        data_path='../data/vico_processed_30fps', 
        meta_data_path='../data/RLD_data.csv', 
        mode='train'
    )
    val_data = ViCoListenerDataset(
        data_path='../data/vico_processed_30fps', 
        meta_data_path='../data/RLD_data.csv', 
        mode='test'
    )
    if not args.distributed:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(train_data))
        dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=args.workers, sampler=data.distributed.DistributedSampler(val_data))
    return dataset

def get_lm_listener_dataloaders(batch_size):
    dataset = {}
    train_data = LmListenerDataset(
        data_path='../data/lm_listener_data/trevorconanstephen/', 
        mode='train'
    )
    val_data = LmListenerDataset(
        data_path='../data/lm_listener_data/trevorconanstephen/', 
        mode='test'
    )
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate_lm)
    dataset["valid"] = data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=pad_collate_lm)
    
    return dataset


if __name__ == "__main__":
    get_dataloaders()
