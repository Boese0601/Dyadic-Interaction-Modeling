import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import config
from base.baseTrainer import load_state_dict
from models import get_model


from x_transformers import TransformerWrapper, ContinuousTransformerWrapper, Encoder, Decoder, AutoregressiveWrapper, ContinuousAutoregressiveWrapper
from x_utils import *

class Transformer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim,
        enc_max_seq_len,
        cross_attn_tokens_dropout=0.,
        **kwargs
    ): 
        # todo tie pre-trained embeddings
        super().__init__()
        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout
        
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        self.encoder = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim, cross_attend = True, **dec_kwargs)
        )
        self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=-100, pad_value=0)

    def forward(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None, listener_ids_decoded=None):

        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)
        if listener_ids_decoded is not None:
            # append listener context to the beginning of the sequence
            listener_ids_decoded = listener_ids_decoded.unsqueeze(1)
            enc = torch.cat([listener_ids_decoded, enc], dim=1)

            # update mask
            mask = torch.cat([torch.ones(mask.shape[0], 1, dtype=torch.bool).cuda(), mask], dim=1)
            # append 2 leading -100 at start of each sequence to ignore first 2 tokens (speaker and listener ids)
            tgt = torch.cat([torch.ones(tgt.shape[0], 1, dtype=torch.long).cuda()*-100, tgt], dim=1)
        
        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        loss, (logits, cache) = self.decoder(tgt, context = enc, context_mask = mask, return_outputs = True)
        if listener_ids_decoded is not None:
            logits = logits[:, 1:, :]
        return loss, logits
        

class SLM(nn.Module):
    def __init__(self):
        super().__init__()
        config_speaker_pth = './config.yaml'
        config_listener_pth = './config.yaml'

        # model_speaker_pth = './runs_vico_pretrain_speaker/model/model.pth.tar'
        # model_listener_pth = './runs_vico_pretrain_listener/model/model.pth.tar'
        model_speaker_pth = './runs_speaker_new/_RANK0/model/model.pth.tar'
        model_listener_pth = './runs/listener_exp/model/model.pth.tar'

        config_speaker = config.load_cfg_from_cfg_file(config_speaker_pth)
        config_listener = config.load_cfg_from_cfg_file(config_listener_pth)

        model_speaker = get_model(config_speaker)
        model_listener = get_model(config_listener)

        checkpoint_speaker = torch.load(model_speaker_pth, map_location=lambda storage, loc: storage.cpu())
        checkpoint_listener = torch.load(model_listener_pth, map_location=lambda storage, loc: storage.cpu())

        load_state_dict(model_speaker, checkpoint_speaker['state_dict'])
        load_state_dict(model_listener, checkpoint_listener['state_dict'])
        print('Load models successfully')
        self.speaker_face_quan_num = config_speaker.face_quan_num
        self.speaker_zquant_dim = config_speaker.zquant_dim

        self.speaker_vq = model_speaker
        self.speaker_vq.eval()
        for param in self.speaker_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.speaker_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.speaker_vq.decoder.parameters():
            param.requires_grad = True
        self.listener_vq = model_listener
        self.listener_vq.eval()
        for param in self.listener_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.listener_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.listener_vq.decoder.parameters():
            param.requires_grad = True
        
        
        dim_in = 56
        dim = 384
        enc_max_seq_len = 2048
        enc_kwargs = {
            'depth': 4,
            'heads': 12,
            'max_seq_len': 2048
        }
        dec_kwargs = {
            'depth': 4,
            'heads': 12,
            'max_seq_len': 2048,
            'num_tokens': 512,
        }
        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.encoder_s = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.encoder_l = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.encoder_joint = ContinuousTransformerWrapper(
            dim_in = dim,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.patch_embed_s = nn.Parameter(torch.zeros(1, 1, dim_in))
        self.patch_embed_l = nn.Parameter(torch.zeros(1, 1, dim_in))

        self.patch_embed_dec_s = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embed_dec_l = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_s, self.norm_l, self.norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

        dim_a = 768
        self.decoder_joint = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim+dim_a, cross_attend = True, **dec_kwargs)
        )
        self.decoder_joint = AutoregressiveWrapper(self.decoder_joint, ignore_index=-100, pad_value=0)
        # self.head_s = nn.Linear(dim+dim_a, 512)
        # self.head_l = nn.Linear(dim+dim_a, 512)
        # self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)


    def random_masking_unstructured(self, x, mask, mask_ratio):
        N, L, D = x.shape
        # get each sample's length based on mask (for each row, count number of True)
        len_keep = torch.sum(mask, dim=1, dtype=torch.int32)
        final_mask = torch.zeros(N, L, dtype=torch.bool).cuda()
        for i in range(N):
            cur_len = len_keep[i]
            # sample mask_ratio indices lower than cur_len
            mask_indices = torch.randperm(cur_len)[:int(cur_len*mask_ratio)]
            # fill mask_indices with True to final_mask
            final_mask[i, :cur_len][mask_indices] = True   
        # mask positions are True, unmask positions are False 
        return final_mask  
    
    def forward_vq(self, v_speaker, v_listener, mask):
        batch_sz, seq_len, _ = v_speaker.shape
        padded_dim = seq_len * self.speaker_face_quan_num
        z_speaker, z_listener = [], []
        for i in range(batch_sz):
            # (1, self.speaker_zquant_dim, self.speaker_face_quan_num*seq_len) => (1, self.speaker_zquant_dim, padded_dim)
            speaker_feats = self.speaker_vq.encode(v_speaker[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_speaker_feats = F.pad(speaker_feats, (0, padded_dim - speaker_feats.shape[-1]), value=0)
            z_speaker.append(padded_speaker_feats)
            listener_feats = self.listener_vq.encode(v_listener[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_listener_feats = F.pad(listener_feats, (0, seq_len - listener_feats.shape[-1]), value=-100)
            z_listener.append(padded_listener_feats)
        z_listener = torch.stack(z_listener, dim=0)
        z_speaker = torch.stack(z_speaker, dim=0)
        return z_speaker, z_listener


    def forward_encoder(self, v_speaker, v_listener, mask, mask_ratio=0.15): 
        v_speaker_cp = v_speaker.clone()
        v_listener_cp = v_listener.clone()

        mask_speaker = self.random_masking_unstructured(v_speaker, mask, mask_ratio)
        mask_listener = self.random_masking_unstructured(v_listener, mask, mask_ratio)

        # add patch_embeds to v_speaker and v_listener but keep masking position zeros
        v_speaker_cp = v_speaker_cp + self.patch_embed_s
        v_listener_cp = v_listener_cp + self.patch_embed_l

        v_speaker_cp[mask_speaker] = 0
        v_listener_cp[mask_listener] = 0

        x_s = self.encoder_s(v_speaker_cp, mask=mask, return_embeddings = True)
        x_l = self.encoder_l(v_listener_cp, mask=mask, return_embeddings = True)
        x_joint = torch.cat([x_s, x_l], dim=1)
        x_joint = self.encoder_joint(x_joint, mask=torch.cat([mask, mask], dim=-1), return_embeddings = True)
        x_l = self.encoder_joint(x_l,  mask=mask, return_embeddings = True)
        x_s = self.encoder_joint(x_s, mask=mask, return_embeddings = True)
        x_joint, x_l, x_s = self.norm(x_joint), self.norm_l(x_l), self.norm_s(x_s)
        return x_s, x_l, x_joint, mask_speaker, mask_listener
    
    def forward_decoder(self, x_s, x_l, z_s, z_l, x_a, mask):
        x_s = x_s + self.patch_embed_dec_s
        x_l = x_l + self.patch_embed_dec_l
        x_s = torch.cat([x_s, x_a], dim=-1)
        x_l = torch.cat([x_l, x_a], dim=-1)
        l_ce_s, (px_s, _) = self.decoder_joint(z_s, context=x_l, context_mask=mask, return_outputs=True)
        l_ce_l, (px_l, _) = self.decoder_joint(z_l, context=x_s, context_mask=mask, return_outputs=True)
        # px_s = self.head_s(self.decoder_joint(x_l, mask=mask, return_embeddings = True))
        # px_l = self.head_l(self.decoder_joint(x_s, mask=mask, return_embeddings = True))
        # px_s = self.head_s(x_l)
        # px_l = self.head_l(x_s)
        # l_ce_s = self.ce_loss(px_s.view(-1, px_s.shape[-1]), z_s.view(-1))
        # l_ce_l = self.ce_loss(px_l.view(-1, px_l.shape[-1]), z_l.view(-1))

        return l_ce_s, l_ce_l, px_s, px_l
    
    def forward_vq_decoder(self, logits_s, logits_l):
        pred_seq_s = torch.argmax(logits_s, dim=-1)
        pred_seq_l = torch.argmax(logits_l, dim=-1)
        min_encodings_s = torch.zeros(pred_seq_s.shape[0], pred_seq_s.shape[1], 512).to(logits_s.device)
        min_encodings_l = torch.zeros(pred_seq_l.shape[0], pred_seq_l.shape[1], 512).to(logits_l.device)
        min_encodings_s.scatter_(2, pred_seq_s.unsqueeze(2), 1)
        min_encodings_l.scatter_(2, pred_seq_l.unsqueeze(2), 1)
        zq_s = torch.matmul(min_encodings_s, self.speaker_vq.quantize.embedding.weight)
        zq_l = torch.matmul(min_encodings_l, self.listener_vq.quantize.embedding.weight)
        zq_s = zq_s.permute(0, 2, 1)
        zq_l = zq_l.permute(0, 2, 1)
        pred_cont_seq_s = self.speaker_vq.decode(zq_s)
        pred_cont_seq_l = self.listener_vq.decode(zq_l)
        return pred_cont_seq_s, pred_cont_seq_l
    
    def forward_continuous_loss(self, pred, target, mask):
        target = target[:, 1:, :]
        mask = mask[:, 1:]
        B = len(target)
        target = target.reshape(B * (target.shape[1]), -1)
        pred = pred.reshape(B * (pred.shape[1]), -1)
        mask = mask.reshape(B * (mask.shape[1]))
        pred_flat = pred[mask]
        target_flat = target[mask]
        pairwise_distance_pose = F.pairwise_distance(pred_flat[:, 0:6], target_flat[:, 0:6])
        pairwise_distance_exp = F.pairwise_distance(pred_flat[:, 6:], target_flat[:, 6:])
        loss_cont = torch.mean(pairwise_distance_exp) + torch.mean(pairwise_distance_pose)
        return loss_cont

    def forward_contrastive(self, s_rep, l_rep, mask, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation
        len_keep = torch.sum(mask, dim=1, dtype=torch.int32)
        # reduce s_rep via average, but use len_keep to get the correct mean
        s_rep_r, l_rep_r = [], []
        for i in range(len(len_keep)):
            s_rep_r.append(torch.mean(s_rep[i, :len_keep[i]], dim=0))
            l_rep_r.append(torch.mean(l_rep[i, :len_keep[i]], dim=0))
        s_rep = torch.stack(s_rep_r, dim=0)
        l_rep = torch.stack(l_rep_r, dim=0)

        s_rep = torch.nn.functional.normalize(s_rep, dim=-1)
        l_rep = torch.nn.functional.normalize(l_rep, dim=-1)

        total = torch.mm(s_rep, torch.transpose(l_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=s_rep.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=s_rep.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=s_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc

    def forward(self, v_speaker, v_listener, v_audio, mask, speaker_ids=None, listener_ids=None, mode='train'):
        with torch.no_grad():
            z_s, z_l = self.forward_vq(v_speaker, v_listener, mask)
        x_s, x_l, x_joint, mask_speaker, mask_listener = self.forward_encoder(v_speaker, v_listener, mask)
        nce, c_acc = self.forward_contrastive(x_s, x_l, mask)
        x_joint_s = x_joint[:, :x_s.shape[1], :]
        x_joint_l = x_joint[:, x_s.shape[1]:, :]
        # mask z_s and z_l so that unmasked positions become -100
        z_s[~mask_speaker] = -100
        z_l[~mask_listener] = -100
        l_ce_s, l_ce_l, px_s, px_l = self.forward_decoder(x_joint_s, x_joint_l, z_s, z_l, v_audio, mask)
        pred_cont_seq_s, pred_cont_seq_l = self.forward_vq_decoder(px_s, px_l)
        l_cont_s = self.forward_continuous_loss(pred_cont_seq_s, v_speaker, mask_speaker)
        l_cont_l = self.forward_continuous_loss(pred_cont_seq_l, v_listener, mask_listener)
        total_loss = l_ce_s + l_ce_l + l_cont_s + l_cont_l + nce
        d = {
            'l_ce_s': l_ce_s,
            'l_ce_l': l_ce_l,
            'l_cont_s': l_cont_s,
            'l_cont_l': l_cont_l,
            'nce': nce,
            'c_acc': c_acc
        }
        return total_loss, d, None
    
class SLMFT(nn.Module):
    def __init__(self):
        super().__init__()
        config_speaker_pth = './config.yaml'
        config_listener_pth = './config.yaml'

        # model_speaker_pth = './runs_vico_pretrain_speaker/model/model.pth.tar'
        # model_listener_pth = './runs_vico_pretrain_listener/model/model.pth.tar'
        model_speaker_pth = './runs_speaker_new/_RANK0/model/model.pth.tar'
        model_listener_pth = './runs/listener_exp/model/model.pth.tar'

        config_speaker = config.load_cfg_from_cfg_file(config_speaker_pth)
        config_listener = config.load_cfg_from_cfg_file(config_listener_pth)

        model_speaker = get_model(config_speaker)
        model_listener = get_model(config_listener)

        checkpoint_speaker = torch.load(model_speaker_pth, map_location=lambda storage, loc: storage.cpu())
        checkpoint_listener = torch.load(model_listener_pth, map_location=lambda storage, loc: storage.cpu())

        load_state_dict(model_speaker, checkpoint_speaker['state_dict'])
        load_state_dict(model_listener, checkpoint_listener['state_dict'])
        print('Load models successfully')
        self.speaker_face_quan_num = config_speaker.face_quan_num
        self.speaker_zquant_dim = config_speaker.zquant_dim

        self.speaker_vq = model_speaker
        self.speaker_vq.eval()
        for param in self.speaker_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.speaker_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.speaker_vq.decoder.parameters():
            param.requires_grad = False
        self.listener_vq = model_listener
        self.listener_vq.eval()
        for param in self.listener_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.listener_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.listener_vq.decoder.parameters():
            param.requires_grad = False
        
        
        dim_in = 56
        dim = 384
        enc_max_seq_len = 2048
        enc_kwargs = {
            'depth': 4,
            'heads': 12,
            'max_seq_len': 2048
        }
        dec_kwargs = {
            'depth': 4,
            'heads': 12,
            'max_seq_len': 2048,
            'num_tokens': 512,
        }
        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', False)

        self.encoder_s = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.encoder_l = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.encoder_joint = ContinuousTransformerWrapper(
            dim_in = dim,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.patch_embed_s = nn.Parameter(torch.zeros(1, 1, dim_in))
        self.patch_embed_l = nn.Parameter(torch.zeros(1, 1, dim_in))

        self.patch_embed_dec_s = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embed_dec_l = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_s, self.norm_l, self.norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

        dim_a = 768
        self.decoder_joint = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim+dim_a, cross_attend = True, **dec_kwargs)
        )
        self.decoder_joint = AutoregressiveWrapper(self.decoder_joint, ignore_index=-100, pad_value=0, mask_prob=0.15)
        # self.decoder_joint = ContinuousTransformerWrapper(
        #     dim_in = dim+dim_a,
        #     dim_out = dim,
        #     max_seq_len = enc_max_seq_len,
        #     attn_layers = Encoder(dim = dim, **enc_kwargs)
        # )
        # self.head_s = nn.Linear(dim, 512)
        # self.head_l = nn.Linear(dim, 512)
        # self.head_s = nn.Linear(dim+dim_a, 512)
        # self.head_l = nn.Linear(dim+dim_a, 512)
        # self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward_encoder(self, v_speaker, mask): 
        v_speaker_cp = v_speaker.clone()

        # add patch_embeds to v_speaker and v_listener but keep masking position zeros
        v_speaker_cp = v_speaker_cp + self.patch_embed_s

        attn_mask = ~torch.triu(torch.ones(v_speaker_cp.shape[1], v_speaker_cp.shape[1]), diagonal=1).bool().cuda()

        x_s = self.encoder_s(v_speaker_cp, mask=mask, attn_mask=attn_mask, return_embeddings = True)
        x_s = self.encoder_joint(x_s, mask=mask, attn_mask=attn_mask, return_embeddings = True)
        x_s = self.norm_s(x_s)
        return x_s
    
    def forward_decoder(self, x_s, z_l, x_a, mask, mode):
        x_s = x_s + self.patch_embed_dec_s
        x_s = torch.cat([x_s, x_a], dim=-1)
        if mode == 'train':
            l_ce_l, (px_l, _) = self.decoder_joint(z_l, context=x_s, context_mask=mask, return_outputs=True)
        else:
            px_l = self.decoder_joint.generate(z_l[:, 0].unsqueeze(1), seq_len=z_l.shape[1]-1, context=x_s, context_mask=mask)
            l_ce_l = 0.0
        return l_ce_l, px_l
    
    def forward_vq_decoder(self, logits_l, mode='train'):
        if mode == 'train':
            pred_seq_l = torch.argmax(logits_l, dim=-1)
        else:
            pred_seq_l = logits_l
        min_encodings_l = torch.zeros(pred_seq_l.shape[0], pred_seq_l.shape[1], 512).to(logits_l.device)
        min_encodings_l.scatter_(2, pred_seq_l.unsqueeze(2), 1)
        zq_l = torch.matmul(min_encodings_l, self.listener_vq.quantize.embedding.weight)
        zq_l = zq_l.permute(0, 2, 1)
        pred_cont_seq_l = self.listener_vq.decode(zq_l)
        return pred_cont_seq_l
    
    def forward_continuous_loss(self, pred, target, mask):
        target = target[:, 1:, :]
        mask = mask[:, 1:]
        B = len(target)
        target = target.reshape(B * (target.shape[1]), -1)
        pred = pred.reshape(B * (pred.shape[1]), -1)
        mask = mask.reshape(B * (mask.shape[1]))
        pred_flat = pred[mask]
        target_flat = target[mask]
        pairwise_distance_pose = F.pairwise_distance(pred_flat[:, 0:6], target_flat[:, 0:6])
        pairwise_distance_exp = F.pairwise_distance(pred_flat[:, 6:], target_flat[:, 6:])
        loss_cont = torch.mean(pairwise_distance_exp) + torch.mean(pairwise_distance_pose)
        return loss_cont
    
    def forward_vq(self, v_speaker, v_listener, mask):
        batch_sz, seq_len, _ = v_speaker.shape
        padded_dim = seq_len * self.speaker_face_quan_num
        z_speaker, z_listener = [], []
        for i in range(batch_sz):
            # (1, self.speaker_zquant_dim, self.speaker_face_quan_num*seq_len) => (1, self.speaker_zquant_dim, padded_dim)
            speaker_feats = self.speaker_vq.encode(v_speaker[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_speaker_feats = F.pad(speaker_feats, (0, padded_dim - speaker_feats.shape[-1]), value=0)
            z_speaker.append(padded_speaker_feats)
            listener_feats = self.listener_vq.encode(v_listener[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_listener_feats = F.pad(listener_feats, (0, seq_len - listener_feats.shape[-1]), value=-100)
            z_listener.append(padded_listener_feats)
        z_listener = torch.stack(z_listener, dim=0)
        z_speaker = torch.stack(z_speaker, dim=0)
        return z_speaker, z_listener

    def forward(self, v_speaker, v_listener, v_audio, mask, mode='train', speaker_ids=None, listener_ids=None):
        with torch.no_grad():
            z_s, z_l = self.forward_vq(v_speaker, v_listener, mask)
            z_l[~mask] = -100
        _, z_l = self.forward_vq(v_speaker, v_listener, mask)
        x_s = self.forward_encoder(v_speaker, mask)
        l_ce_l, px_l = self.forward_decoder(x_s, z_l, v_audio, mask, mode=mode)
        pred_cont_seq_l = self.forward_vq_decoder(px_l, mode=mode)
        l_cont_l = self.forward_continuous_loss(pred_cont_seq_l, v_listener, mask)
        total_loss = l_ce_l + l_cont_l
        d = {
            'l_ce_s': 0,
            'l_ce_l': l_ce_l,
            'l_cont_s': 0,
            'l_cont_l': l_cont_l,
            'nce': 0,
            'c_acc': 0
        }
        return total_loss, d, pred_cont_seq_l

class SpeakerSLMFT(nn.Module):
    def __init__(self):
        super().__init__()
        config_speaker_pth = './config.yaml'
        config_listener_pth = './config.yaml'

        model_speaker_pth = './runs_speaker_new/_RANK0/model/model.pth.tar'
        model_listener_pth = './runs/listener_exp/model/model.pth.tar'

        config_speaker = config.load_cfg_from_cfg_file(config_speaker_pth)
        config_listener = config.load_cfg_from_cfg_file(config_listener_pth)

        model_speaker = get_model(config_speaker)
        model_listener = get_model(config_listener)

        checkpoint_speaker = torch.load(model_speaker_pth, map_location=lambda storage, loc: storage.cpu())
        checkpoint_listener = torch.load(model_listener_pth, map_location=lambda storage, loc: storage.cpu())

        load_state_dict(model_speaker, checkpoint_speaker['state_dict'])
        load_state_dict(model_listener, checkpoint_listener['state_dict'])
        print('Load models successfully')
        self.speaker_face_quan_num = config_speaker.face_quan_num
        self.speaker_zquant_dim = config_speaker.zquant_dim

        self.listener_vq = model_listener
        self.listener_vq.eval()
        for param in self.listener_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.listener_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.listener_vq.decoder.parameters():
            param.requires_grad = False

        converter = EmocaConverter()
        model_converter_path = './best_converter.pt'
        checkpoint_convert = torch.load(model_converter_path, map_location=lambda storage, loc: storage.cpu())
        converter.load_state_dict(checkpoint_convert)

        self.speaker_vq = model_speaker
        self.speaker_vq.eval()
        for param in self.speaker_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.speaker_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.speaker_vq.decoder.parameters():
            param.requires_grad = True

        self.vertice_mapping = converter.vertice_mapping
        self.squasher = converter.squasher
        self.vertice_map_reverse = converter.vertice_map_reverse
        self.vertice_map_reverse_lstm = converter.vertice_map_reverse_lstm
        self.vertice_map_reverse_lstm_2 = converter.vertice_map_reverse_lstm_2
        self.vertice_map_reverse2 = converter.vertice_map_reverse2

        for p in self.vertice_mapping.parameters():
            p.requires_grad = False
        for p in self.squasher.parameters():
            p.requires_grad = False
        
        dim_in = 56
        dim = 384
        enc_max_seq_len = 2048
        enc_kwargs = {
            'depth': 4,
            'heads': 12,
            'max_seq_len': 2048
        }
        dec_kwargs = {
            'depth': 4,
            'heads': 12,
            'max_seq_len': 2048,
            'num_tokens': 512,
        }
        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.encoder_s = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.encoder_l = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.encoder_joint = ContinuousTransformerWrapper(
            dim_in = dim,
            dim_out = dim,
            max_seq_len = enc_max_seq_len,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )
        self.patch_embed_s = nn.Parameter(torch.zeros(1, 1, dim_in))
        self.patch_embed_l = nn.Parameter(torch.zeros(1, 1, dim_in))

        self.patch_embed_dec_s = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embed_dec_l = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_s, self.norm_l, self.norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

        dim_a = 768
        self.decoder_joint = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim+dim_a, cross_attend = True, **dec_kwargs)
        )
        self.decoder_joint = AutoregressiveWrapper(self.decoder_joint, ignore_index=-100, pad_value=0)
        self.mse_loss = nn.MSELoss()
        self.scaling = torch.FloatTensor([10**0]).cuda()
        with open("../data/CodeTalker/BIWI/regions/lve.txt") as f:
            maps = f.read().split(", ")
            mouth_map = [int(i) for i in maps]
        self.mouth_map = mouth_map
        self.mouth_scaling = torch.FloatTensor([10**0]).cuda()

        self.W = torch.nn.Parameter(torch.randn(2))
        self.W.requires_grad = True
        self.speaker_embed = nn.Embedding(15, 384)


    def forward_encoder(self, v_speaker, mask): 
        v_speaker_cp = v_speaker.clone()

        # add patch_embeds to v_speaker and v_listener but keep masking position zeros
        v_speaker_cp = v_speaker_cp + self.patch_embed_s


        x_s = self.encoder_s(v_speaker_cp, mask=mask, return_embeddings = True)
        x_s = self.encoder_joint(x_s, mask=mask, return_embeddings = True)
        x_s = self.norm_s(x_s)
        return x_s
    
    def forward_decoder(self, x_l, z_s, x_a, mask, mode='train'):
        x_l = x_l + self.patch_embed_dec_l
        x_l = torch.cat([x_l, x_a], dim=-1)
        if mode == 'train':
            l_ce_s, (px_s, _) = self.decoder_joint(z_s, context=x_l, context_mask=mask, return_outputs=True)
        else:
            px_s = self.decoder_joint.generate(z_s[:, 0].unsqueeze(1), seq_len=z_s.shape[1]-1, context=x_l, context_mask=mask)
            l_ce_s = 0.0
        return l_ce_s, px_s
    
    def forward_vq_decoder(self, logits_s, type='emoca', mode='train'):
        if mode == 'train':
            pred_seq_s = torch.argmax(logits_s, dim=-1)
        else:
            pred_seq_s = logits_s
        min_encodings_s = torch.zeros(pred_seq_s.shape[0], pred_seq_s.shape[1], 512).to(logits_s.device)
        min_encodings_s.scatter_(2, pred_seq_s.unsqueeze(2), 1)
        zq_s = torch.matmul(min_encodings_s, self.speaker_vq.quantize.embedding.weight)
        zq_s = zq_s.permute(0, 2, 1)
        pred_cont_seq_s_emoca = self.speaker_vq.decode(zq_s)
        if type == 'emoca':
            pred_cont_seq_s, _ = self.vertice_map_reverse_lstm(pred_cont_seq_s_emoca)
            pred_cont_seq_s = self.vertice_map_reverse(pred_cont_seq_s)
        else:
            pred_cont_seq_s, _ = self.vertice_map_reverse_lstm_2(pred_cont_seq_s_emoca)
            pred_cont_seq_s = self.vertice_map_reverse2(pred_cont_seq_s)
        return pred_cont_seq_s, pred_cont_seq_s_emoca
    
    def forward_continuous_loss(self, pred, target, mask):
        target = target[:, 1:, :]
        mask = mask[:, 1:]
        B = len(target)
        target = target.reshape(B * (target.shape[1]), -1)
        pred = pred.reshape(B * (pred.shape[1]), -1)
        mask = mask.reshape(B * (mask.shape[1]))
        pred_flat = pred[mask]
        target_flat = target[mask]
        pairwise_distance_pose = F.pairwise_distance(pred_flat[:, 0:6], target_flat[:, 0:6])
        pairwise_distance_exp = F.pairwise_distance(pred_flat[:, 6:], target_flat[:, 6:])
        loss_cont = torch.mean(pairwise_distance_exp) + torch.mean(pairwise_distance_pose)
        return loss_cont
    
    def forward_vq(self, v_speaker, v_listener, mask):
        batch_sz, seq_len, _ = v_speaker.shape
        padded_dim = seq_len * self.speaker_face_quan_num
        z_speaker, z_listener = [], []
        for i in range(batch_sz):
            # (1, self.speaker_zquant_dim, self.speaker_face_quan_num*seq_len) => (1, self.speaker_zquant_dim, padded_dim)
            speaker_feats = self.speaker_vq.encode(v_speaker[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_speaker_feats = F.pad(speaker_feats, (0, padded_dim - speaker_feats.shape[-1]), value=0)
            z_speaker.append(padded_speaker_feats)
            listener_feats = self.listener_vq.encode(v_listener[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_listener_feats = F.pad(listener_feats, (0, seq_len - listener_feats.shape[-1]), value=-100)
            z_listener.append(padded_listener_feats)
        z_listener = torch.stack(z_listener, dim=0)
        z_speaker = torch.stack(z_speaker, dim=0)
        return z_speaker, z_listener

    def forward(self, v_speaker, v_speaker_emoca, v_audio, mask, template, mode='train', speaker_ids=None):
        template = template.unsqueeze(1)
        v_speaker_orig = v_speaker.clone()
        v_speaker = v_speaker - template
        v_speaker = self.vertice_mapping(v_speaker)
        v_speaker = self.squasher(v_speaker.permute(0,2,1)).permute(0,2,1)
        with torch.no_grad():
            z_s, z_s_emoca = self.forward_vq(v_speaker, v_speaker_emoca, mask)
        # z_s = z_s.detach()
        # z_l = z_l.detach()
        # x_s = self.forward_encoder(v_speaker, mask)
        if speaker_ids is None:
            x_l = torch.zeros(v_audio.shape[0], v_audio.shape[1], 384).cuda()
        else:
            spk_embed = self.speaker_embed(speaker_ids)
            x_l = spk_embed.unsqueeze(1).repeat(1, v_audio.shape[1], 1)
        l_ce_s, px_s = self.forward_decoder(x_l, z_s_emoca, v_audio, mask, mode=mode)
        # l_ce_s2, px_s2 = self.forward_decoder(x_l, z_s, v_audio, mask, mode=mode)

        pred_cont_seq_s, pred_cont_seq_s_emoca = self.forward_vq_decoder(px_s, type='emoca', mode=mode)
        # pred_cont_seq_s2, _ = self.forward_vq_decoder(px_s2, type='mesh', mode=mode)

        pred_cont_seq_s = pred_cont_seq_s + template
        # pred_cont_seq_s2 = pred_cont_seq_s2 + template

        l_cont_s = self.mse_loss(pred_cont_seq_s, v_speaker_orig[:, 1:, :]) 
        v_speaker_orig_mouth = v_speaker_orig.reshape(-1, 23370, 3)[:, self.mouth_map, :].reshape(-1, 4996*3)
        pred_cont_seq_s_mouth = pred_cont_seq_s.reshape(-1, 23370, 3)[:, self.mouth_map, :].reshape(-1, 4996*3)
        # pred_cont_seq_s_mouth2 = pred_cont_seq_s2.reshape(-1, 23370, 3)[:, self.mouth_map, :].reshape(-1, 4996*3)

        l_mouth_loss = self.mse_loss(pred_cont_seq_s_mouth, v_speaker_orig_mouth[1:, :]) 
        l_emoca_loss = self.mse_loss(pred_cont_seq_s_emoca, v_speaker_emoca[:, 1:, :])
        
        # softmax W [2x1] then merge pred_cont_seq_s and pred_cont_seq_s2
        # softmax_w = F.softmax(self.W, dim=0)
        # final_pred_cont_seq_s = softmax_w[0] * pred_cont_seq_s + softmax_w[1] * pred_cont_seq_s2
        
        # l_cont_s = self.scaling * l_cont_s + self.mouth_scaling * l_mouth_loss
        # l_cont_s = self.scaling * l_cont_s + self.scaling * l_mouth_loss
        l_cont_s = l_emoca_loss 
        total_loss = l_ce_s + l_cont_s 
        d = {
            'l_ce_s': 0,
            'l_ce_l': l_ce_s,
            'l_cont_s': self.mouth_scaling * l_mouth_loss,
            'l_cont_l': l_cont_s,
            'nce': 0,
            'c_acc': 0
        }
        return total_loss, d, pred_cont_seq_s_emoca
    
class EmocaConverter(nn.Module):
    def __init__(self):
        super().__init__()
        config_speaker_pth = './config.yaml'
        model_speaker_pth = './runs_speaker_new/_RANK0/model/model.pth.tar'

        config_speaker = config.load_cfg_from_cfg_file(config_speaker_pth)

        model_speaker = get_model(config_speaker)

        checkpoint_speaker = torch.load(model_speaker_pth, map_location=lambda storage, loc: storage.cpu())

        load_state_dict(model_speaker, checkpoint_speaker['state_dict'])
        print('Load models successfully')
        self.speaker_face_quan_num = config_speaker.face_quan_num
        self.speaker_zquant_dim = config_speaker.zquant_dim

        self.speaker_vq = model_speaker
        self.speaker_vq.eval()
        for param in self.speaker_vq.parameters():
            param.requires_grad = False

        size = 70110
        dim = 56

        self.vertice_mapping = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(0.2, True))
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.InstanceNorm1d(dim, affine=False)
                    )]
        self.squasher = nn.Sequential(*layers)
        # self.vertice_map_reverse = nn.Linear(dim,size, bias=False)
        # self.vertice_map_reverse = nn.Sequential(
        #     nn.Linear(dim, 384),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(384, 768),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(768, 70110)
        # )
        # bi-directional lstm
        self.vertice_map_reverse_lstm = nn.LSTM(
            input_size=dim,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.vertice_map_reverse_lstm_2 = nn.LSTM(
            input_size=dim,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.vertice_map_reverse = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 70110)
        )
        self.vertice_map_reverse2 = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, True),
            nn.Linear(768, 70110)
        )
        self.criterion = nn.MSELoss()

    def forward(self, inputs, template, v_speaker):
        template = template.unsqueeze(1) #B,V*3 -> B, 1, V*3
        # inputs = inputs - template
        # inputs = self.vertice_mapping(inputs)
        # inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]
        dec, _, _ = self.speaker_vq(v_speaker)
        # outputs = self.vertice_map_reverse(dec)
        outputs, _ = self.vertice_map_reverse_lstm(dec)
        outputs = self.vertice_map_reverse(outputs)
        outputs = outputs + template

        # dec2, _, _ = self.speaker_vq(inputs)
        # outputs2, _ = self.vertice_map_reverse_lstm_2(dec2)
        # outputs2 = self.vertice_map_reverse2(outputs2)
        # outputs2 = outputs2 + template
        return outputs, None
    
if __name__=='__main__':  
    model = SpeakerSLMFT().cuda()
    v_speaker = torch.randn(8, 27, 70110).cuda()
    template = torch.randn(8, 70110).cuda()
    v_listener = torch.randn(8, 27, 56).cuda()
    v_audio = torch.randn(8, 27, 768).cuda()
    mask = torch.ones(8, 27, dtype=torch.bool).cuda()
    total_loss, d, pred_cont_seq_s = model(v_speaker, v_listener, v_audio, mask, template)
    print(d)
    # model = EmocaConverter().cuda()
    # inputs = torch.randn(8, 138, 70110).cuda()
    # template = torch.randn(8, 70110).cuda()
    # outputs = model(inputs, template)
    print('here')
    # z_listener = model.generate(v_speaker)
    # print(z_listener, z_listener.shape)
    
    # x = torch.randn(4, 27, 56)
    # mask = torch.zeros(4, 27, dtype=torch.bool)
    # mask[0, 0:5] = True
    # mask[1, 0:10] = True
    # mask[2, 0:15] = True
    # mask[3, 0:20] = True
    # random_masking_unstructured(x, mask, 0.75)
