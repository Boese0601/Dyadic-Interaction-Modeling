import os
#import cv2 
import lmdb
import math
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
####
torch._six = torch
######
from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from util.distributed import master_only_print as print
from data.vox_video_dataset import VoxVideoDataset
from config import Config

class Dataset_img_feat(torch.utils.data.Dataset):
    def __init__(self, opt, is_inference=False):
        self.opt = opt 
        self.feat_dir = opt['feat_dir']
        self.fst_dir = opt['fst_dir']
        self.out_dir = opt['out_dir']
        file_ending = ".1.png"
        self.nams = [{"nam" : f[:-len(file_ending)],
                      "img_src" : os.path.join(self.fst_dir, f) } for f in os.listdir(self.fst_dir) if f.endswith(file_ending)]
        for n in self.nams:
            n['feat'] = os.path.join(self.feat_dir, n['nam'])
        self.samples = []
        for n in self.nams:
            img_src_data = Image.open(n['img_src'])
            #Normalize
            img_src_data = F.normalize(F.to_tensor(img_src_data), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #img_src_data = F.to_tensor(img_src_data)
            for fram_idx in sorted(os.listdir(n['feat'])):
                exp_feat = np.load(os.path.join(n['feat'], fram_idx, "exp.npy"))
                pose_feat = np.load(os.path.join(n['feat'], fram_idx, "pose.npy"))
                ####concat_feat = np.concatenate([pose_feat, exp_feat], axis=0) #cur_feat = np.concatenate((vals['posecode'][j].detach().cpu().numpy(), vals['expcode'][j].detach().cpu().numpy()), axis=0)
                concat_feat = np.concatenate([exp_feat,  [0,0], pose_feat], axis=0)
                #semantic = [semantic[:, 6:], np.zeros((semantic.shape[0], 2)), semantic[:, : 6]]
                concat_feat = np.stack([concat_feat]*27, axis=0)
                concat_feat = torch.Tensor(concat_feat).permute(1,0)
                out_dir = os.path.join(self.out_dir, "fake", n['nam'], fram_idx + ".png")
                out_dir_warp = os.path.join(self.out_dir, "warp", n['nam'], fram_idx + ".png")
                self.samples.append({"img_src" : img_src_data, 
                                     "feat" : concat_feat, 
                                     "out_dir" : out_dir, 
                                     "out_dir_warp" : out_dir_warp})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
    def transform_semantic1(self, semantic, frame_index, num_frames):
        semantic = [i for i in semantic.values()]
        index = self.obtain_seq_index(frame_index, num_frames)
        semantic =  np.stack(semantic, axis=0)
        self.b = semantic, index    
        coeff_3dmm = semantic[index,...]
        coeff_3dmm = coeff_3dmm
        return torch.Tensor(coeff_3dmm).permute(1,0)
#
import pickle
class Dataset_img_feat_Diformat(torch.utils.data.Dataset):
    def __init__(self, opt, is_inference=False):
        self.opt = opt 
        self.semantic_radius = 1
        self.feat_dir = opt['feat_dir']
        self.fst_dir = opt['fst_dir']
        self.out_dir = opt['out_dir']
        file_ending = ".1.png"
        self.nams = []
        conv = {"52": "0039vi", "71" : "0045vid_" , "72" : "0059vid", "all" : ""}
        for of in os.listdir(self.fst_dir):
            for f in os.listdir(self.feat_dir):
                if f[:4].startswith(of[:4]) and conv[opt['finetune_num']] in f:
                    self.nams.append({"nam" : of,
                                  "img_src" : os.path.join(self.fst_dir, of),
                                  "feat" : os.path.join(self.feat_dir, f) })
        self.samples = []
        for n in self.nams:
            img_src_data = Image.open(n['img_src'])
            #Normalize
            img_src_data = F.normalize(F.to_tensor(img_src_data), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #img_src_data = F.to_tensor(img_src_data)
            with open(n['feat'], 'rb') as f:
                feat_all = pickle.load(f)
            keys_feat = sorted(list(feat_all.keys()))
            for fram_idx in range(len(keys_feat)):
                feat = feat_all
                ####concat_feat = np.concatenate([pose_feat, exp_feat], axis=0) #cur_feat = np.concatenate((vals['posecode'][j].detach().cpu().numpy(), vals['expcode'][j].detach().cpu().numpy()), axis=0)
                concat_feat = self.transform_semantic1(feat, fram_idx, len(feat_all))
                out_dir = os.path.join(self.out_dir, "fake", n['nam'], keys_feat[fram_idx] + ".png")
                out_dir_warp = os.path.join(self.out_dir, "warp", n['nam'], keys_feat[fram_idx] + ".png")
                self.samples.append({"img_src" : img_src_data, 
                                     "feat" : concat_feat, 
                                     "out_dir" : out_dir, 
                                     "out_dir_warp" : out_dir_warp})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
    def transform_semantic1(self, semantic, frame_index, num_frames):
        semantic = [i[1] for i in sorted(semantic.items())] #[i for i in semantic.values()]
        index = self.obtain_seq_index(frame_index, num_frames)
        ############################################################################################################################################print(index)
        semantic =  np.stack(semantic, axis=0)
        if 1 or self.opt.decapirender:
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

class Dataset_img_feat_personspecific(torch.utils.data.Dataset):
    def __init__(self, opt, is_inference=False, allowed_nams = None):
        self.opt = opt 
        self.feat_dir = opt['feat_dir']
        self.fst_dir = opt['fst_dir']
        self.out_dir = opt['out_dir']
        file_ending = ".png"
        #"nam" is with .0 or .1; feat is without
        self.nams = [{"nam" : f[:-len(file_ending)],
                      "img_src" : os.path.join(self.fst_dir, f) } for f in os.listdir(self.fst_dir) \
                        if f.endswith(file_ending)  ]
        if allowed_nams is not None:
            #import pdb; pdb.set_trace()
            ########self.nams = [n for n in self.nams if n['nam'][:-2] in allowed_nams]
            self.nams = [n for n in self.nams if n['nam'] in allowed_nams]
            #self.nams = [n for n in self.nams if os.path.basename(n['img_src'])[:-len(".1.png")] in allowed_nams]

        for n in self.nams:
            n['feat'] = os.path.join(self.feat_dir, n['nam'][: -len(".0")])
        #import pdb; pdb.set_trace()
        self.nams = [n for n in self.nams if os.path.exists(n['feat'])]
        self.samples = []
        print(self.nams)
        for n in self.nams:
            img_src_data = Image.open(n['img_src'])
            #Normalize
            img_src_data = F.normalize(F.to_tensor(img_src_data), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #img_src_data = F.to_tensor(img_src_data)
            for fram_idx in sorted(os.listdir(n['feat'])):
                exp_feat = np.load(os.path.join(n['feat'], fram_idx, "exp.npy"))
                pose_feat = np.load(os.path.join(n['feat'], fram_idx, "pose.npy"))
                ####concat_feat = np.concatenate([pose_feat, exp_feat], axis=0) #cur_feat = np.concatenate((vals['posecode'][j].detach().cpu().numpy(), vals['expcode'][j].detach().cpu().numpy()), axis=0)
                concat_feat = np.concatenate([exp_feat,  [0,0], pose_feat], axis=0)
                #semantic = [semantic[:, 6:], np.zeros((semantic.shape[0], 2)), semantic[:, : 6]]
                concat_feat = np.stack([concat_feat]*27, axis=0)
                concat_feat = torch.Tensor(concat_feat).permute(1,0)
                if os.path.exists(os.path.join(self.out_dir, "fake", n['nam'])):
                    print("Skipping: ", os.path.join(self.out_dir, "fake", n['nam']))
                    continue
                out_dir = os.path.join(self.out_dir, "fake", n['nam'], fram_idx + ".png")
                out_dir_warp = os.path.join(self.out_dir, "warp", n['nam'], fram_idx + ".png")
                self.samples.append({"img_src" : img_src_data, 
                                     "feat" : concat_feat, 
                                     "out_dir" : out_dir, 
                                     "out_dir_warp" : out_dir_warp})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
    def transform_semantic1(self, semantic, frame_index, num_frames):
        semantic = [i for i in semantic.values()]
        index = self.obtain_seq_index(frame_index, num_frames)
        semantic =  np.stack(semantic, axis=0)
        self.b = semantic, index    
        coeff_3dmm = semantic[index,...]
        coeff_3dmm = coeff_3dmm
        return torch.Tensor(coeff_3dmm).permute(1,0)

class Dataset_img_feat_personspecificLM(torch.utils.data.Dataset):
    def __init__(self, opt, is_inference=False, allowed_nams = None):
        self.opt = opt 
        self.feat_dir = opt['feat_dir']
        self.fst_dir = opt['fst_dir']
        self.out_dir = opt['out_dir']
        file_ending = ".png"
        self.nams = [{"nam" : f[:-len(file_ending)],
                      "img_src" : os.path.join(self.fst_dir, f) } for f in os.listdir(self.fst_dir) \
                        if f.endswith(file_ending)  ]
        #if allowed_nams is not None:
            #import pdb; pdb.set_trace()
            #self.nams = [n for n in self.nams if n['nam'] in allowed_nams]#[:-2]
            #self.nams = [n for n in self.nams if os.path.basename(n['img_src'])[:-len(".1.png")] in allowed_nams]
        for n in self.nams:
            n['feat'] = os.path.join(self.feat_dir, n['nam'])
        self.nams = [n for n in self.nams if os.path.exists(n['feat'])]
        self.samples = []
        print(self.nams)
        for n in self.nams:
            img_src_data = Image.open(n['img_src'])
            #Normalize
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                transforms.Resize((256,256))# TODO: remove
            ])     
            img_src_data = self.transform(img_src_data)#F.Resize((256,256))(F.normalize(F.to_tensor(img_src_data), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
            #img_src_data = F.to_tensor(img_src_data)
            for fram_idx in sorted(os.listdir(n['feat'])):
                exp_feat = np.load(os.path.join(n['feat'], fram_idx, "exp.npy"))
                pose_feat = np.load(os.path.join(n['feat'], fram_idx, "pose.npy"))
                ####concat_feat = np.concatenate([pose_feat, exp_feat], axis=0) #cur_feat = np.concatenate((vals['posecode'][j].detach().cpu().numpy(), vals['expcode'][j].detach().cpu().numpy()), axis=0)
                concat_feat = np.concatenate([exp_feat,  [0,0], pose_feat], axis=0)
                #semantic = [semantic[:, 6:], np.zeros((semantic.shape[0], 2)), semantic[:, : 6]]
                concat_feat = np.stack([concat_feat]*27, axis=0)
                concat_feat = torch.Tensor(concat_feat).permute(1,0)
                if os.path.exists(os.path.join(self.out_dir, "fake", n['nam'])):
                    print("Skipping: ", os.path.join(self.out_dir, "fake", n['nam']))
                    continue
                out_dir = os.path.join(self.out_dir, "fake", n['nam'], fram_idx + ".png")
                out_dir_warp = os.path.join(self.out_dir, "warp", n['nam'], fram_idx + ".png")
                self.samples.append({"img_src" : img_src_data, 
                                     "feat" : concat_feat, 
                                     "out_dir" : out_dir, 
                                     "out_dir_warp" : out_dir_warp})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
    def transform_semantic1(self, semantic, frame_index, num_frames):
        semantic = [i for i in semantic.values()]
        index = self.obtain_seq_index(frame_index, num_frames)
        semantic =  np.stack(semantic, axis=0)
        self.b = semantic, index    
        coeff_3dmm = semantic[index,...]
        coeff_3dmm = coeff_3dmm
        return torch.Tensor(coeff_3dmm).permute(1,0)


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--cross_id', action='store_true')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--feat_mode', type=int, default=0)
    parser.add_argument('--finetune_num', type=int, default=-1)
    parser.add_argument('--datanam', type=str, default='vico')
    parser.add_argument('--customfeatdir', type=str, default=None)

    args = parser.parse_args()
    return args

def get_dict():
    import pandas as pd
    # Load the dataset
    file_path = ./datasets/vico/RLD_data.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    # Filter the dataset for the 'test' data split and specified listener IDs
    ##specified_listener_ids = [71, 72, 52, 15, 54, 5, 19, 87, 0, 21, 89,]
    specified_listener_ids = [0, 5, 15, 19, 21, 52, 54, 71, 72, 87, 15, 52, 54, 71, 72, 87, 89, ]
    filtered_df = df[(df['data_split'] == 'test') & (df['listener_id'].isin(specified_listener_ids))]
    # Create a dictionary where each listener ID maps to a list of corresponding "audio" entries
    audio_dict = filtered_df.groupby('listener_id')['listener'].apply(list).to_dict()
    return audio_dict
def get_dict_all():# 
    # 0, 21, 33, 48, 52, 54, 71, 72, 75, 78, 85, 86, 87, 88, 8 # is all ids in val
    import pandas as pd
    # Load the dataset
    file_path = ./datasets/vico/RLD_data.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    # Filter the dataset for the 'test' data split and specified listener IDs
    ##specified_listener_ids = [71, 72, 52, 15, 54, 5, 19, 87, 0, 21, 89,]
    filtered_df = df[(df['data_split'] == 'test') ]
    # Create a dictionary where each listener ID maps to a list of corresponding "audio" entries
    audio_dict = filtered_df.groupby('listener_id')['listener'].apply(list).to_dict()
    return audio_dict



def get_dict_precomp():
    audio_dict = {0: ['Ew2z_sYABE0_000063_000083'],
         21: ['z5x1Nx2eiuQ_000111_000142', 'z5x1Nx2eiuQ_000177_000196'],
         52: ['rSM89vRNq4Y_000661_000669', 'rSM89vRNq4Y_001469_001473'],
         54: ['rSM89vRNq4Y_000689_000699'],
         71: ['UCA1A5GqCdQ_000318_000327',
          'UCA1A5GqCdQ_000353_000361',
          'UCA1A5GqCdQ_000690_000696',
          'UCA1A5GqCdQ_001058_001061',
          'UCA1A5GqCdQ_001113_001119',
          'UCA1A5GqCdQ_003373_003382',
          'UCA1A5GqCdQ_003690_003700',
          'UCA1A5GqCdQ_004221_004229',
          'UCA1A5GqCdQ_004879_004893'],
         72: ['bPiofmZGb8o_002969_002986',
          'bPiofmZGb8o_004758_004767',
          'UCA1A5GqCdQ_000610_000619',
          'UCA1A5GqCdQ_000718_000726',
          'UCA1A5GqCdQ_000774_000787',
          'UCA1A5GqCdQ_001154_001162',
          'UCA1A5GqCdQ_001213_001223',
          'UCA1A5GqCdQ_001301_001308',
          'UCA1A5GqCdQ_002627_002636',
          'UCA1A5GqCdQ_003785_003803',
          'UCA1A5GqCdQ_004124_004130',
          'UCA1A5GqCdQ_004996_005013',
          'UCA1A5GqCdQ_005365_005373'],
         87: ['yTInFHb_o1Q_000516_000551'],
         89: ['yTInFHb_o1Q_000473_000485',
  'yTInFHb_o1Q_000506_000516',
  'yTInFHb_o1Q_000551_000558'],
 48: ['YFhkBtIUgBM_000087_000091', 'YFhkBtIUgBM_000106_000107'],
 88: ['yTInFHb_o1Q_000506_000516',
  'yTInFHb_o1Q_000516_000551',
  'yTInFHb_o1Q_000551_000558']}
    return audio_dict

#Deadlinestu./datasets/vico/EMOCPIRender_our/bPiofmZGb8o_002969_002986/0/exp.npy
#Deadlinestu./datasets/vico/framsfst/0jhYr0nhSAE_000053_000059.0.png

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)
    opt.device = "cuda:0"
    # create a model
    net_G, net_G_ema, opt_G, sch_G = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, \
                          opt_G, sch_G, None)
    #opt.logdir = "result/emoca1/"
    #ELP eccv22 l2l_vico_candor_smooth
    opt1 = {"feat_dir" : f"./datasets/vico/rebuttal/{args.datanam}/",
            "fst_dir" : "./datasets/vico/framsfst/",
            "out_dir" : f"./datasets/vico/{args.datanam}/"}
    if args.customfeatdir is not None:
        opt1['feat_dir'] = args.customfeatdir
    #opt1 = {"feat_dir" : ./datasets/vico/New_ourmodel_emocapred/",
    #        "fst_dir" : ./datasets/vico/framsfst/",
    #        "out_dir" : ./datasets/vico/NewEMOCPIRender_our_pirender/"}
    opt1_______________LM_ = {"feat_dir" : "./datasets/vico/New_ourmodel_emocapred/",
            "fst_dir" : ."/datasets/vico/framsfst/",
            "out_dir" : ."/datasets/vico/NewEMOCPIRender_our_TRAINTESTpirender/"}
    if args.finetune_num == -1:
        allowed_nams = None
    else:
        opt1['finetune_num'] = args.finetune_num 
        ################################################################################################################allowed_nams = get_dict_precomp()[opt1['finetune_num']]
        allowed_nams = get_dict_all()[opt1['finetune_num']]

    #opt.logdir = f"result/vicofinetuning_{opt1['finetune_num']}" #"result/face2/"  №opt.logdir =  "result/TRAINEDemoca/"
    #opt.logdir = f"result/LMlistfinetuning_clean"
    opt.logdir = f"result/VICOfinetuningl_{opt1['finetune_num']}"
    opt.logdir______________ = f"result/VICOtestintrainfinetuningl_{opt1['finetune_num']}"
    
    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter, del_map = False)  
                          
    net_G = trainer.net_G_ema.eval()
    #dataset = VoxVideoDataset(opt.data, is_inference=True)

    opt.dataset_type = 2 
    if opt.dataset_type == 0:
        dataset = Dataset_img_feat(opt1)
    elif opt.dataset_type == 1:
        dataset = Dataset_img_feat_Diformat(opt1)
    elif opt.dataset_type == 2:
        print("allowed_nams: ", allowed_nams)
        dataset = Dataset_img_feat_personspecific(opt1, allowed_nams = allowed_nams)
    elif opt.dataset_type == 3:
        dataset = Dataset_img_feat_personspecificLM(opt1, allowed_nams = allowed_nams)
    import pdb; pdb.set_trace()
    with torch.no_grad():
        for data in dataset:
            input_source = data['img_src'].cuda().unsqueeze(0)
            output_images, gt_images, warp_images = [],[],[]
            target_semantic = data['feat'].cuda().unsqueeze(0) #[0]???
            output_dict = net_G(input_source, target_semantic)
            img = (output_dict['fake_image'].cpu().clamp_(-1, 1) + 1) / 2
            warp_img = (output_dict['warp_image'].cpu().clamp_(-1, 1) + 1) / 2
            print("Saving images to: ", data['out_dir'])
            os.makedirs(os.path.dirname(data['out_dir']), exist_ok=True)
            os.makedirs(os.path.dirname(data['out_dir_warp']), exist_ok=True)
            img_pil = to_pil_image((img[0]))
            img_pil.save(data['out_dir'])
            warp_img_pil = to_pil_image(warp_img[0])
            warp_img_pil.save(data['out_dir_warp'])
            

"""
def write2video(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i]) 

    out_name = results_dir+'.mp4' 
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release() """


