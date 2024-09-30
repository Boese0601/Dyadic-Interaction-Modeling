import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class VoxDataset_LM(Dataset):#LM_Dataset dataset
    def __init__(self, opt, is_inference):
        """
        mode_split -- 2 for vico. 0 for train, 1 for 20 samples of vox in val, 9 for 1 sample of vox in val
        """
        print("LMdataset entered")
        self.opt = opt
        self.minimal_sample_distance = opt.minimal_sample_distance
        self.path_vidsdir = opt.path
        self.path_feat = opt.path_feat
        self.semantic_radius = opt.semantic_radius
        all_feats = list(sorted(os.listdir(self.path_feat)))
        if opt.mode_split == 2:
            self.person_ids = [i[:-len(".pkl")] for i in all_feats]
        if opt.mode_split == 2:
            #import pandas as pd
            #markup = opt.markup_file
            #markup = pd.read_excel(markup,engine='openpyxl')
            #if not is_inference:
            #    markup = markup[markup['data_split'] == 'train']
            #else:
            #    markup = markup[markup['data_split'] == 'test']
            self.lst = []
            if opt.mode_split == 2:
                cur_set_pers= sorted(list(set([i for i in self.person_ids])))
            else:
                train_set_pers = sorted(list(set([i for i in self.person_ids ])))
            self.person_ids = sorted(list(set(self.person_ids)))
        if opt.mode_split == 2:
            self.person_ids = [i for i in self.person_ids if i in cur_set_pers]
        else:
            if is_inference:  
                self.person_ids = [i for i in self.person_ids if not i in train_set_pers]
            else:
                self.person_ids = [i for i in self.person_ids if i in train_set_pers]
        #import pdb; pdb.set_trace()
        self.pers2feats = {}
        for person_id in self.person_ids:
            person_feats = [i for i in all_feats if i.startswith(person_id)]
            self.pers2feats[person_id] = person_feats
        self.pers2vids = {}
        all_vids = list(sorted(os.listdir(self.path_vidsdir)))
        for person_id in self.person_ids:
            videos = [i for i in all_vids if i.startswith(person_id)]
            self.pers2vids[person_id] = videos
        self.person_ids = self.person_ids * opt.multiplier  #multiplier (original 100) makes 4M iters

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                transforms.Resize((256,256))# TODO: remove
            ])     
        self.person_ids = self.person_ids      
    def feat_2_framedir(self, feat_name):
        if self.opt.mode_split == 2:
            frame_dir =  os.path.join(self.path_vidsdir, feat_name[:-len(".pkl")])
        else:
            frame_dir =  os.path.join(self.path_vidsdir, feat_name[:-len(".pkl")])
        return frame_dir
    
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        
        return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data={}
        # person_id is 100x num of videos; but contains people
        # chose 1 person
        #import pdb; pdb.set_trace()
        person_id = self.person_ids[index]
        #get one of videos with this person
        feat_item = random.choices(self.pers2feats[person_id], k=1)[0]
        #
        fil_nam = os.path.join(self.path_feat, feat_item)
        data['feat_all_path'] = fil_nam
        with open(fil_nam, 'rb') as f:
            coeff = pickle.load(f)
        #        
        frame_dir = self.feat_2_framedir(feat_item)
        
        # select frame idxs
        cur_framdir_lst = list(sorted(os.listdir(frame_dir)))
        num_frames = len(cur_framdir_lst)
        frame_source, frame_target = self.random_select_frames(num_frames, r = self.minimal_sample_distance)
        #############################
        
            
        #Prepare frames with PIL
        
        data['source_path'] = os.path.join(frame_dir, cur_framdir_lst[frame_source])
        img1 = Image.open(data['source_path'])
        data['source_image'] = self.transform(img1)

        data['target_path'] = os.path.join(frame_dir, cur_framdir_lst[frame_target])
        img2 = Image.open(data['target_path'])
        data['target_image'] = self.transform(img2) 

        # Prepare semantics
        data['target_semantics'] = self.transform_semantic1(coeff, frame_target, num_frames)
        data['source_semantics'] = self.transform_semantic1(coeff, frame_source, num_frames)
    
        return data
    def random_select_frames(self, num_frames, r = 40):
        first_idx = random.choice(list(range(num_frames)))
        valid_starts = list(range(max(0, first_idx - r))) + list(range(min(num_frames, first_idx + r + 1), num_frames))
        second_idx = random.choice(valid_starts)
        return first_idx, second_idx
    
    def random_select_frames_no_limits(self, num_frames):
        frame_idx = random.choices(list(range(num_frames)), k=2)
        return frame_idx[0], frame_idx[1]

    def transform_semantic1(self, semantic, frame_index, num_frames):
        semantic = [i[1] for i in sorted(semantic.items())] #[i for i in semantic.values()]
        index = self.obtain_seq_index(frame_index, num_frames)
        #########################
        semantic =  np.stack(semantic, axis=0)
        if self.opt.decapirender:
            semantic = [semantic[:, 6:], np.zeros((semantic.shape[0], 2)), semantic[:, : 6]]
        else:
            semantic = [semantic[:, 6:], semantic[:, : 6]]
        semantic = np.concatenate(semantic, axis=1)

        self.b = semantic, index    
        coeff_3dmm = semantic[index,...]
        if self.semantic_radius == 1:
            coeff_3dmm = np.concatenate([coeff_3dmm, ] * (13*2+1), axis=0)

        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        # Get ids seq around index  (5,2) -> [3,4,5,6,7]  
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [min(max(item, 0), num_frames-1) for item in seq ]
        return seq



class VoxDataset(Dataset):#VICO dataset and pretraining VoxCeleb dataset depending on path and path_feat
    def __init__(self, opt, is_inference):
        """
        mode_split -- 2 for vico. 0 for train, 1 for 20 samples of vox in val, 9 for 1 sample of vox in val
        """
        print(f'VoxDataset init')
        #opt.path = "./dataset/vids_train/"
        self.opt = opt
        self.minimal_sample_distance = opt.minimal_sample_distance
        ####"./dataset/demo_vids/"
        self.path_vidsdir = opt.path
        self.path_feat = opt.path_feat
        self.semantic_radius = opt.semantic_radius
        all_feats = list(sorted(os.listdir(self.path_feat)))
        if opt.mode_split == 2:
            self.person_ids = [i[:-len(".pkl")] for i in all_feats]
        else:
            self.person_ids = [i[:-len("_00006.pkl")] for i in all_feats]
        if opt.mode_split == 2:
            pass
        else:
            self.person_ids = [i.split('_')[0] for i in self.person_ids]
        if opt.mode_split == 0:
            train_set_pers = sorted(list(set(self.person_ids)))[:-20]# 50k if multipl 100
        elif opt.mode_split == 1:
            train_set_pers = sorted(list(set(self.person_ids)))[:-1]
        elif opt.mode_split == 2:
            import pandas as pd
            markup = opt.markup_file
            markup = pd.read_excel(markup,engine='openpyxl')
            if not is_inference:
                if opt.testintrain:
                    markup = markup[:]
                else:
                    markup = markup[markup['data_split'] == 'train']
            else:
                markup = markup[markup['data_split'] == 'test']
            if opt.person_number == -1: # -1 to use all
                listener_list = markup[:].iloc[:, 2].tolist()
                speaker_list = markup[:].iloc[:, 3].tolist()
            else:
                listener_list = markup[markup['listener_id'] == opt.person_number].iloc[:, 2].tolist()
                speaker_list = markup[markup['speaker_id'] == opt.person_number].iloc[:, 3].tolist()
            markup = listener_list 
            if opt.include_speakersintrain:
                markup += speaker_list
            
            self.lst = []
            if opt.mode_split == 2:
                cur_set_pers= sorted(list(set([i for i in self.person_ids if i in markup])))
            else:
                train_set_pers = sorted(list(set([i for i in self.person_ids if i in markup])))
            self.person_ids = sorted(list(set(self.person_ids)))
        if opt.mode_split == 2:
            self.person_ids = [i for i in self.person_ids if i in cur_set_pers]
        else:
            if is_inference:  
                self.person_ids = [i for i in self.person_ids if not i in train_set_pers]
            else:
                self.person_ids = [i for i in self.person_ids if i in train_set_pers]
        #import pdb; pdb.set_trace()
        self.pers2feats = {}
        for person_id in self.person_ids:
            person_feats = [i for i in all_feats if i.startswith(person_id)]
            self.pers2feats[person_id] = person_feats
        self.pers2vids = {}
        all_vids = list(sorted(os.listdir(self.path_vidsdir)))
        for person_id in self.person_ids:
            videos = [i for i in all_vids if i.startswith(person_id)]
            self.pers2vids[person_id] = videos
        self.person_ids = self.person_ids * opt.multiplier  #multiplier (original 100) makes 4M iters

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                transforms.Resize((256,256))# TODO: remove
            ])     
        self.person_ids = self.person_ids      
        #import pdb; pdb.set_trace()
    def feat_2_framedir(self, feat_name):
        if self.opt.mode_split == 2:
            frame_dir =  os.path.join(self.path_vidsdir, "vid_vico_videos_" + feat_name[:-len(".pkl")])
        else:
            frame_dir =  os.path.join(self.path_vidsdir, feat_name[:-len(".pkl")])
        return frame_dir
    
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        
        return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data={}
        # person_id is 100x num of videos; but contains people
        # chose 1 person
        #import pdb; pdb.set_trace()
        person_id = self.person_ids[index]
        #get one of videos with this person
        feat_item = random.choices(self.pers2feats[person_id], k=1)[0]
        #
        fil_nam = os.path.join(self.path_feat, feat_item)
        data['feat_all_path'] = fil_nam
        with open(fil_nam, 'rb') as f:
            coeff = pickle.load(f)#######FOR NEW PIRENDER ONLY #TODO: CHECK CORRESPONDACE. As in new pirender pretrained on exp[52],pose[6]; here pose[6], exp[50]#coeff = [coeff_pre[:, 6:], np.zeros((coeff_pre.shape[0], 2)), coeff_pre[:, : 6]]#coeff = np.concatenate(coeff, axis=1)
        #        
        frame_dir = self.feat_2_framedir(feat_item)
        #print(feat_item, frame_dir)
        # select frame idxs
        cur_framdir_lst = list(sorted(os.listdir(frame_dir)))
        num_frames = len(cur_framdir_lst)
        frame_source, frame_target = self.random_select_frames(num_frames, r = self.minimal_sample_distance)
        ##########################################################print(os.path.join(self.path_feat, feat_item))
        
            
        #Prepare frames with PIL
        
        data['source_path'] = os.path.join(frame_dir, cur_framdir_lst[frame_source])
        img1 = Image.open(data['source_path'])
        data['source_image'] = self.transform(img1)

        data['target_path'] = os.path.join(frame_dir, cur_framdir_lst[frame_target])
        img2 = Image.open(data['target_path'])
        data['target_image'] = self.transform(img2) 

        # Prepare semantics
        data['target_semantics'] = self.transform_semantic1(coeff, frame_target, num_frames)
        data['source_semantics'] = self.transform_semantic1(coeff, frame_source, num_frames)
        
        return data
    def random_select_frames(self, num_frames, r = 40):
        first_idx = random.choice(list(range(num_frames)))
        valid_starts = list(range(max(0, first_idx - r))) + list(range(min(num_frames, first_idx + r + 1), num_frames))
        second_idx = random.choice(valid_starts)
        return first_idx, second_idx
    
    def random_select_frames_no_limits(self, num_frames):
        frame_idx = random.choices(list(range(num_frames)), k=2)
        return frame_idx[0], frame_idx[1]

    def transform_semantic1(self, semantic, frame_index, num_frames):
        semantic = [i[1] for i in sorted(semantic.items())] #[i for i in semantic.values()]
        index = self.obtain_seq_index(frame_index, num_frames)
        ############################################################################################################################################print(index)
        semantic =  np.stack(semantic, axis=0)
        if self.opt.decapirender:
            semantic = [semantic[:, 6:], np.zeros((semantic.shape[0], 2)), semantic[:, : 6]]#TODO: CHECK CORRESPONDACE. As in new pirender pretrained on exp[52],pose[6]; here pose[6], exp[50]
        else:
            semantic = [semantic[:, 6:], semantic[:, : 6]]#TODO: CHECK CORRESPONDACE. As in new pirender pretrained on exp[52],pose[6]; here pose[6], exp[50]
        semantic = np.concatenate(semantic, axis=1)

        self.b = semantic, index    
        coeff_3dmm = semantic[index,...]
        if self.semantic_radius == 1:
            coeff_3dmm = np.concatenate([coeff_3dmm, ] * (13*2+1), axis=0)

        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        # Get ids seq around index  (5,2) -> [3,4,5,6,7]  
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [min(max(item, 0), num_frames-1) for item in seq ]
        return seq

    

class VoxDataset_old(Dataset): # Legacy
    def __init__(self, opt, is_inference):
        print(f'VoxDataset old used. Warning')
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path, str(opt.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_list.txt" if is_inference else "train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]

        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.person_ids = self.person_ids * 100

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))

        return video_items, person_ids            

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        
        return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data={}
        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        frame_source, frame_target = self.random_select_frames(video_item)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], frame_source)
            img_bytes_1 = txn.get(key) 
            key = format_for_lmdb(video_item['video_name'], frame_target)
            img_bytes_2 = txn.get(key) 
            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))

        img1 = Image.open(BytesIO(img_bytes_1))
        data['source_image'] = self.transform(img1)

        img2 = Image.open(BytesIO(img_bytes_2))
        data['target_image'] = self.transform(img2) 

        data['target_semantics'] = self.transform_semantic(semantics_numpy, frame_target)
        data['source_semantics'] = self.transform_semantic(semantics_numpy, frame_source)
    
        return data
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq

