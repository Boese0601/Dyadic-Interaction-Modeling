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
        
    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs):
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = mask, **kwargs)
    
class ContinuousTransformer(nn.Module):
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

        dec_transformer_kwargs = pick_and_pop(['max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.decoder = ContinuousTransformerWrapper(
            dim_in = dim,
            dim_out = 56,
            max_seq_len = enc_max_seq_len,
            attn_layers = Decoder(dim = dim, **dec_kwargs)
        )
        self.decoder = ContinuousAutoregressiveWrapper(self.decoder, ignore_index=-100, pad_value=0)

    def forward(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None):

        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)
        
        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        loss = self.decoder(x=enc, tgt=tgt, mask=mask)
        return loss
    
    def generate(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None):
        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)

        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        pred = self.decoder.generate(x=enc, mask=mask)
        return pred

    
class ListenerGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        config_speaker_pth = './config_speaker_old.yaml'
        config_listener_pth = './config.yaml'

        model_speaker_pth = './runs/speaker_exp/model/model.pth.tar'
        model_listener_pth = './runs/listener_exp/model/model.pth.tar'
        # model_speaker_pth = './runs_speaker_raw/model/model.pth.tar'
        # model_listener_pth = './runs_listener_raw/model/model.pth.tar'

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
        for param in self.speaker_vq.parameters():
            param.requires_grad = False
        self.listener_vq = model_listener
        self.listener_vq.eval()
        for param in self.listener_vq.quantize.parameters():
            param.requires_grad = False
        for param in self.listener_vq.encoder.parameters():
            param.requires_grad = False
        for param in self.listener_vq.decoder.parameters():
            param.requires_grad = True
        
        self.generator = Transformer(
            dim_in = self.speaker_face_quan_num * self.speaker_zquant_dim,
            # dim_in = 768+56,
            dim = 512,
            enc_depth = 6,
            enc_heads = 8,
            enc_max_seq_len = 1024,
            dec_num_tokens = 512,
            dec_depth = 6,
            dec_heads = 8,
            dec_max_seq_len = 1024,
        )
        self.speaker_embeddings = nn.Embedding(100, 256)
        self.listener_embeddings = nn.Embedding(100, 256)
        self.fc_speaker = nn.Linear(256, 1024)
        self.fc_listener = nn.Linear(256, 512)

    def get_3dmm_loss(self, pred, gt):
        b, t, c = pred.shape
        xpred = pred.reshape(b * t, c)
        xgt = gt.reshape(b * t, c)
        pairwise_distance = F.pairwise_distance(xpred, xgt)
        loss = torch.mean(pairwise_distance)
        spiky_loss = self.get_spiky_loss(pred, gt)
        return loss, spiky_loss
    
    def get_spiky_loss(self, pred, gt):
        b, t, c = pred.shape
        pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]
        gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]
        pred_spiky = pred_spiky.reshape(b * (t - 1), c)
        gt_spiky = gt_spiky.reshape(b * (t - 1), c)
        pairwise_distance = F.pairwise_distance(pred_spiky, gt_spiky)
        return torch.mean(pairwise_distance)

    def forward(self, v_speaker, v_listener, mask, speaker_ids=None, listener_ids=None):
        batch_sz, seq_len, _ = v_speaker.shape
        padded_dim = seq_len * self.speaker_face_quan_num
        x_speaker, z_listener = [], []
        for i in range(batch_sz):
            # (1, self.speaker_zquant_dim, self.speaker_face_quan_num*seq_len) => (1, self.speaker_zquant_dim, padded_dim)
            speaker_feats = self.speaker_vq.encode(v_speaker[i, :, :][mask[i]].unsqueeze(0))[0]
            padded_speaker_feats = F.pad(speaker_feats, (0, padded_dim - speaker_feats.shape[-1]), value=0)
            x_speaker.append(padded_speaker_feats)
            listener_feats = self.listener_vq.encode(v_listener[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_listener_feats = F.pad(listener_feats, (0, seq_len - listener_feats.shape[-1]), value=-100)
            z_listener.append(padded_listener_feats)

        x_speaker = torch.cat(x_speaker, dim=0)
        x_speaker = x_speaker.view(x_speaker.shape[0], -1, self.speaker_face_quan_num, self.speaker_zquant_dim).contiguous()
        x_speaker = x_speaker.view(x_speaker.shape[0], -1, self.speaker_face_quan_num*self.speaker_zquant_dim).contiguous()
        if speaker_ids is not None:
            # decode speaker ids, then concat as first vector of x_speaker
            speaker_ids_decode = self.fc_speaker(F.relu(self.speaker_embeddings(speaker_ids)))
            x_speaker = torch.cat([speaker_ids_decode.unsqueeze(1), x_speaker], dim=1)
            mask_updated = torch.cat([torch.ones(batch_sz, 1, dtype=torch.bool).cuda(), mask], dim=1)
        else:
            mask_updated = mask
        if listener_ids is not None:
            listener_ids_decode = self.fc_listener(F.relu(self.listener_embeddings(listener_ids)))
        else:
            listener_ids_decode = None

        z_listener = torch.stack(z_listener, dim=0)
        loss, logits = self.generator(src=x_speaker, tgt=z_listener, mask=mask_updated, listener_ids_decoded=listener_ids_decode)

        pred_seq = torch.argmax(logits, dim=-1)
        min_encodings = torch.zeros(pred_seq.shape[0], pred_seq.shape[1], 512).to(v_speaker.device)
        # min_encodings.scatter_(1, pred_seq, 1)
        min_encodings.scatter_(2, pred_seq.unsqueeze(2), 1)
        zq = torch.matmul(min_encodings, self.listener_vq.quantize.embedding.weight)
        zq = zq.permute(0, 2, 1)
        pred_cont_seq = self.listener_vq.decode(zq)
        target_cont_seq = v_listener[:, 1: , :]
        # use mask to only select True positions
        pred_cont_seq_flat = pred_cont_seq.reshape(batch_sz*(seq_len-1), -1)
        target_cont_seq_flat = target_cont_seq.reshape(batch_sz*(seq_len-1), -1)
        mask = mask[:, 1:].reshape(batch_sz*(seq_len-1))
        pred_cont_seq_flat = pred_cont_seq_flat[mask]
        target_cont_seq_flat = target_cont_seq_flat[mask]
        # loss_cont_head, loss_cont_head_spiky = self.get_3dmm_loss(pred_cont_seq[:, :, 0:6], target_cont_seq[:, :, 0:6]) 
        # loss_cont_exp, loss_cont_exp_spiky = self.get_3dmm_loss(pred_cont_seq[:, :, 6:], target_cont_seq[:, :, 6:])
        # loss_cont = loss_cont_head + loss_cont_exp + loss_cont_head_spiky + loss_cont_exp_spiky*0.001
        pairwise_distance_pose = F.pairwise_distance(pred_cont_seq_flat[:, 0:6], target_cont_seq_flat[:, 0:6])
        pairwise_distance_exp = F.pairwise_distance(pred_cont_seq_flat[:, 6:], target_cont_seq_flat[:, 6:])
        loss_cont = torch.mean(pairwise_distance_exp) + torch.mean(pairwise_distance_pose)
        loss = loss + loss_cont
        return loss, pred_cont_seq
    
    def generate(self, v_speaker, v_listener, mask):
        batch_sz, seq_len, _ = v_speaker.shape
        padded_dim = seq_len * self.speaker_face_quan_num
        x_speaker, z_listener = [], []
        for i in range(batch_sz):
            # (1, self.speaker_zquant_dim, self.speaker_face_quan_num*seq_len) => (1, self.speaker_zquant_dim, padded_dim)
            speaker_feats = self.speaker_vq.encode(v_speaker[i, :, :][mask[i]].unsqueeze(0))[0]
            padded_speaker_feats = F.pad(speaker_feats, (0, padded_dim - speaker_feats.shape[-1]), value=0)
            x_speaker.append(padded_speaker_feats)
            listener_feats = self.listener_vq.encode(v_listener[i, :, :][mask[i]].unsqueeze(0))[2][2].squeeze()
            padded_listener_feats = F.pad(listener_feats, (0, seq_len - listener_feats.shape[-1]), value=-100)
            z_listener.append(padded_listener_feats)

        x_speaker = torch.cat(x_speaker, dim=0)
        x_speaker = x_speaker.view(x_speaker.shape[0], -1, self.speaker_face_quan_num, self.speaker_zquant_dim).contiguous()
        x_speaker = x_speaker.view(x_speaker.shape[0], -1, self.speaker_face_quan_num*self.speaker_zquant_dim).contiguous()
        z_listener = torch.stack(z_listener, dim=0)
        # torch.zeros(batch_sz, 1, dtype=torch.long).cuda() z_listener[:, 0].unsqueeze(1)
        z_listener_pred = self.generator.generate(
            seq_in=x_speaker, 
            seq_out_start=z_listener[:, 0].unsqueeze(1),
            seq_len=seq_len, 
            mask=mask
        )
        return z_listener_pred, z_listener

class SimpleLSTM(nn.Module):
    def __init__(self, dim_in, dim_agg, V, H, N): 
        super().__init__()
        self.model = nn.LSTM(
            input_size = 56+768,
            hidden_size = 256,
            num_layers = 3,
            batch_first = True,
            bidirectional = True
        )
        self.fc = nn.Linear(256*2, 56)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, x_target, mask=None):
        x, _ = self.model(x)
        out = self.fc(x)
        loss = self.loss_fn(out, x_target)
        return loss, out
    
if __name__=='__main__':        
    # model = ContinuousTransformer(
    #     dim_in=768+56,
    #     dim=512,
    #     enc_depth = 6,
    #     enc_heads = 8,
    #     enc_max_seq_len = 1024,
    #     dec_depth = 6,
    #     dec_heads = 8,
    #     dec_max_seq_len = 1024,
    # )
    model = ListenerGenerator()
    v_speaker = torch.randn(8, 27, 768+56)
    v_listener = torch.randn(8, 27, 56)
    mask = torch.ones(8, 27, dtype=torch.bool)
    loss, logits = model(v_speaker, v_listener, mask)
    print(loss)
    print('here')
    # z_listener = model.generate(v_speaker)
    # print(z_listener, z_listener.shape)
        

