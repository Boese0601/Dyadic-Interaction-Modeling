import os
import torch
import torch.nn as nn
from dataset.data_loader import get_candor_dataloaders, get_vico_dataloaders, get_lm_listener_dataloaders
from x_engine import train_epoch, evaluate_epoch, train_continuous_epoch, evaluate_continuous_epoch
from seq2seq import ListenerGenerator, ContinuousTransformer, SimpleLSTM

from tqdm import tqdm
from metrics.eval_utils import calculate_activation_statistics, calculate_frechet_distance, calculate_variance, calcuate_sid, sts
import numpy as np
from piq import FID
import torch.distributed as dist
import builtins
import pickle5 as pickle

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_dist():
    env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }

    ngpus_per_node = torch.cuda.device_count()

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="nccl")
    crank = int(env_dict['RANK'])

    if(crank!=0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print(f'Init succesfully rank {crank}')
    return crank

# crank = initialize_dist()
crank = 0
device = torch.device("cuda:{}".format(crank))
# model = SimpleLSTM(
#     56+768, 512, 64, 128, 3
# ).to(device)
model = ListenerGenerator().to(device)
# model = ContinuousTransformer(
#     dim_in=768+56,
#     dim=512,
#     enc_depth = 6,
#     enc_heads = 8,
#     enc_max_seq_len = 1024,
#     dec_depth = 6,
#     dec_heads = 8,
#     dec_max_seq_len = 1024,
# ).to(device)
# pretrained_ckpt = 'saved_checkpoints_wcont_candor/epoch_5.pth'

# model = nn.DataParallel(model)

# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[crank], find_unused_parameters=True)
model = nn.DataParallel(model)
pretrained_ckpt = 'saved_checkpoints_wcont_candor/epoch_5.pth'
# pretrained_ckpt = 'saved_checkpoints_personalized_step1/best_model.pth'
model.load_state_dict(torch.load(pretrained_ckpt), strict=False)
# model = model.module

# for p in model.module.parameters():
#     p.requires_grad = False

# unfreze speaker embeddings/fc
# for p in model.module.speaker_embeddings.parameters():
#     p.requires_grad = True
# for p in model.module.fc_speaker.parameters():
#     p.requires_grad = True
# # unfreeze listener embeddings/fc
# for p in model.module.listener_embeddings.parameters():
#     p.requires_grad = True
# for p in model.module.fc_listener.parameters():
#     p.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = None

# dataset = get_candor_dataloaders(batch_size=32)
dataset = get_vico_dataloaders(batch_size=4)
# dataset = get_lm_listener_dataloaders(batch_size=32)
train_loader = dataset['train']
val_loader = dataset['valid']

# num_epochs = 10000//len(train_loader)
num_epochs = 100
best_ppl = 10000
print(f'training for {num_epochs} epochs')
for epoch in range(num_epochs):
    model.train()
    # train_loader.sampler.set_epoch(epoch)
    train_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=0.0, print_freq=200, epoch=epoch)
    # train_continuous_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=0.0, print_freq=200, epoch=epoch)
    # if crank == 0:
    #     torch.save(model.state_dict(), os.path.join('saved_checkpoints_wcont_candor', f'epoch_{epoch}.pth'))

    # if epoch != num_epochs-1:
    #     continue
    model.eval()
    gt, pred = [], []
    x = []
    all_fnames = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            src, tgt, src_len, _, fnames = batch
            src = src.to(device)
            tgt = tgt.to(device)
            mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
            for j in range(src.shape[0]):
                mask[j, :src_len[j]] = True
            loss, _ = model(src, tgt, mask)
            # cont_pred = model.module.generate(src, tgt, mask)
            z_pred, z_gt = model.module.generate(src, tgt, mask)
            for j in range(z_pred.shape[0]):
                # pred_seq = logits[j][mask[j, 1:]].unsqueeze(0).cpu()
                # pred_seq = torch.argmax(pred_seq, dim=-1)
                pred_seq = z_pred[j][mask[j]].cpu()
                pred_seq = pred_seq.squeeze().unsqueeze(1)
                min_encodings = torch.zeros(pred_seq.shape[0], 512)
                min_encodings.scatter_(1, pred_seq, 1)
                zq = torch.matmul(min_encodings.cuda(), model.module.listener_vq.quantize.embedding.weight)
                zq = zq.unsqueeze(0)
                zq = zq.permute(0, 2, 1)
                pred_cont_seq = model.module.listener_vq.decode(zq)[0].cpu().numpy()
                gt_seq = tgt[j][1:][mask[j, 1:]].cpu().numpy()

                # gt_seq = tgt[j][mask[j]][1:].cpu().numpy()
                # pred.append(cont_pred[j][mask[j, 1:]].cpu().numpy())
                # gt.append(gt_seq)
                # x.append(src[j][1:][mask[j, 1:]][:, 0:56].cpu().numpy())
                # all_fnames.append(fnames[j])
                pred.append(pred_cont_seq)
                gt.append(tgt[j][mask[j]].squeeze().cpu().numpy())
                x.append(src[j][mask[j]][:, 0:56].cpu().numpy())

    # gt_dict, pred_dict, x_dict = {}, {}, {}
    # with torch.no_grad():
    #     for i, batch in enumerate(tqdm(val_loader)):
    #         src, tgt, src_len, _, filenames = batch
    #         src = src.to(device)
    #         tgt = tgt.to(device)
    #         mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool).to(device)
    #         for j in range(src.shape[0]):
    #             mask[j, :src_len[j]] = True
    #         loss, cont_pred = model(src, tgt, mask)
    #         # cont_pred = model.module.generate(src, tgt, mask)
    #         # z_pred, z_gt = model.module.generate(src, tgt, mask)
    #         for j in range(cont_pred.shape[0]):
    #             cur_fname = filenames[j]
    #             if cur_fname not in gt_dict.keys():
    #                 gt_dict[cur_fname] = []
    #                 pred_dict[cur_fname] = []
    #                 x_dict[cur_fname] = []
    #             gt_seq = tgt[j][mask[j]][1:].cpu().numpy()
    #             pred_dict[cur_fname].append(cont_pred[j][mask[j, 1:]].cpu().numpy())
    #             gt_dict[cur_fname].append(gt_seq)
    #             x_dict[cur_fname].append(src[j][1:][mask[j, 1:]][:, 0:56].cpu().numpy())

    #             # pred_dict[cur_fname].append(cont_pred[j][mask[j]].cpu().numpy())
    #             # gt_dict[cur_fname].append(tgt[j][mask[j]].cpu().numpy())
    #             # x_dict[cur_fname].append(src[j][mask[j]][:, 0:56].cpu().numpy())

    # output = {}
    # for key in gt_dict:
    #     output[key] = {}
    #     output[key]['gt'] = np.concatenate(gt_dict[key], axis=0)
    #     output[key]['pred'] = np.concatenate(pred_dict[key], axis=0)
    #     output[key]['x'] = np.concatenate(x_dict[key], axis=0)

    # gt, pred, x = [], [], []
    # for key in gt_dict:
    #     gt.append(np.concatenate(gt_dict[key], axis=0))
    #     pred.append(np.concatenate(pred_dict[key], axis=0))
    #     x.append(np.concatenate(x_dict[key], axis=0))

        # output = {}
        # for i in range(len(gt)):
        #     output[all_fnames[i]] = {}
        #     output[all_fnames[i]]['gt'] = gt[i]
        #     output[all_fnames[i]]['pred'] = pred[i]
        #     output[all_fnames[i]]['x'] = x[i]

    fids = []
    for i in range(len(gt)):
        mu1, sigma1 = calculate_activation_statistics(gt[i][:, 0:6])
        mu2, sigma2 = calculate_activation_statistics(pred[i][:, 0:6])
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fids.append(fid)
    print('fid_pose: ', np.mean(fids))

    fids = []
    for i in range(len(gt)):
        mu1, sigma1 = calculate_activation_statistics(gt[i][:, 6:])
        mu2, sigma2 = calculate_activation_statistics(pred[i][:, 6:])
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fids.append(fid)
    print('fid_exp: ', np.mean(fids))

    pfids = []
    for i in range(len(gt)):
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i][:,0:6], gt[i][:,0:6]], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i][:,0:6], pred[i][:,0:6]], axis=-1))
        fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        pfids.append(fid2)
    print('pfid_pose: ', np.mean(pfids))

    pfids = []
    for i in range(len(gt)):
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([x[i][:,6:], gt[i][:,6:]], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([x[i][:,6:], pred[i][:,6:]], axis=-1))
        fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        pfids.append(fid2)
    print('pfid_exp: ', np.mean(pfids))

    total_l2 = []
    for i in range(len(gt)):
        mse = np.mean((gt[i][:, 0:6] - pred[i][:, 0:6])**2)
        total_l2.append(mse)
    print('mse_pose: ', np.mean(total_l2))

    total_l2 = []
    for i in range(len(gt)):
        mse = np.mean((gt[i][:, 6:] - pred[i][:, 6:])**2)
        total_l2.append(mse)
    print('mse_exp: ', np.mean(total_l2))

    sid_pose = calcuate_sid(gt, pred, type='pose')
    sid_pose_gt = calcuate_sid(gt, gt, type='pose')
    print('sid_pose: ', sid_pose, sid_pose_gt)

    sid_exp = calcuate_sid(gt, pred, type='exp')
    sid_exp_gt = calcuate_sid(gt, gt, type='exp')
    print('sid_exp: ', sid_exp, sid_exp_gt)

    gt = np.concatenate(gt, axis=0)
    gt = gt.reshape(-1, 56)
    pred = np.concatenate(pred, axis=0)
    pred = pred.reshape(-1, 56)
    print('var_pose: ', np.var(gt[:, 0:6].reshape(-1, )), np.var(pred[:, 0:6].reshape(-1, )))
    print('var_exp: ', np.var(gt[:, 6:].reshape(-1, )), np.var(pred[:, 6:].reshape(-1, )))

    x = np.concatenate(x, axis=0)
    x = x[:, 0:56]
    pcc_xy_pose = np.corrcoef(gt[:, 0:6].reshape(-1, ), x[:, 0:6].reshape(-1, ))[0, 1]
    pcc_xy_exp = np.corrcoef(gt[:, 6:].reshape(-1, ), x[:, 6:].reshape(-1, ))[0, 1]
    pcc_xypred_pose = np.corrcoef(pred[:, 0:6].reshape(-1, ), x[:, 0:6].reshape(-1, ))[0, 1]
    pcc_xypred_exp = np.corrcoef(pred[:, 6:].reshape(-1, ), x[:, 6:].reshape(-1, ))[0, 1]
    # print('pcc pose: ', pcc_xy_pose, pcc_xypred_pose)
    # print('pcc exp: ', pcc_xy_exp, pcc_xypred_exp)
    print('rpcc pose: ', abs(pcc_xy_pose-pcc_xypred_pose))
    print('rpcc exp: ', abs(pcc_xy_exp-pcc_xypred_exp))


    sts_pose = sts(gt[:, 0:6], pred[:, 0:6])
    sts_exp = sts(gt[:, 6:], pred[:, 6:])
    print('sts pose: ', sts_pose)
    print('sts exp: ', sts_exp)

    # save model
    # torch.save(model.state_dict(), os.path.join('saved_checkpoints_personalized_step1', 'best_model.pth'))

    # with open('output.pkl', 'wb') as f:
    #     pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

        # val_ppl = evaluate_epoch(model, val_loader, device)
        # if np.mean(fids) < best_ppl:
        #     best_ppl = np.mean(fids)
        #     torch.save(model.state_dict(), os.path.join('saved_checkpoints_vico', 'best_model30fps_nopretrain.pth'))
    # print('best ppl: ', best_ppl)

    # best_loss = float('inf')
    # print(f'training for {num_epochs} epochs')
    # for epoch in range(num_epochs):
    #     train_continuous_epoch(model, train_loader, optimizer, device, scheduler=scheduler, clip=0.0, print_freq=200, epoch=epoch)
    #     val_loss = evaluate_continuous_epoch(model, val_loader, device)
    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         torch.save(model.state_dict(), os.path.join('saved_checkpoints_continuous_vico', 'best_model.pth'))
    # print('best loss: ', best_loss)


    # best_ppl without pretrain: 41.47
    # best_ppl with pretrain: 28.3

    # best loss no_pretrain continuous: 0.5455437004566193