import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from conformer import *
from conformer.encoder import *
from conformer.activation import GLU

# For conventional pickers
from obspy.signal.trigger import classic_sta_lta, trigger_onset

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000, return_vec=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.return_vec = return_vec

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if not self.return_vec: 
            # x: (batch_size*num_windows, window_size, input_dim)
            x = x[:] + self.pe.squeeze()

            return self.dropout(x)
        else:
            return self.pe.squeeze()

class cross_attn(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, dropout=0.1):
        super(cross_attn, self).__init__()
        
        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, nhead * d_v, bias=False)
        self.fc = nn.Linear(nhead * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, mean=None, std=None):
        d_k, d_v, nhead = self.d_k, self.d_v, self.nhead
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, nhead, d_k)
        k = self.w_ks(k).view(sz_b, len_k, nhead, d_k)
        v = self.w_vs(v).view(sz_b, len_v, nhead, d_k)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        
        if mean is not None and std is not None:
            mean = mean.unsqueeze(1).unsqueeze(1)
            std = std.unsqueeze(1).unsqueeze(1)
            attn = torch.matmul(q, k.transpose(-2, -1)) * mean + std
            attn = attn / d_k**0.5
        else:
            attn = torch.matmul(q / d_k**0.5, k.transpose(-2, -1))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        output = self.dropout(self.fc(output))
        output += residual
        
        output = self.layer_norm(output)
        
        return output

class cross_attn_layer(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, conformer_class, d_ffn):
        super(cross_attn_layer, self).__init__()
        
        self.cross_attn = cross_attn(nhead=nhead, d_k=d_k, d_v=d_v, d_model=d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn),
                                    nn.ReLU(),
                                    nn.Linear(d_ffn, d_model),
                                )   
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        
        # d_model: dimension of query vector
        self.proj = False
        if d_model != conformer_class:
            self.proj = True
            self.projector = nn.Conv1d(d_model, conformer_class, kernel_size=3, padding='same')
            
    def forward(self, q, k, v, mean=None, std=None):
        out_attn = self.cross_attn(q, k, v, mean=mean, std=std)
            
        out = self.layer_norm(self.ffn(out_attn) + out_attn)
        out = self.dropout(out)
        
        if self.proj:
            out = self.projector(out.permute(0,2,1)).permute(0,2,1)
        return out

class GRADUATE(nn.Module):
    def __init__(self, conformer_class, d_ffn, nhead, d_model, enc_layers, dec_layers,  
                 rep_KV=True, label_type='all', recover_type="conv", wavelength=3000,
                 max_freq=12, stft_recovertype='conv', dualDomain_type='concat', ablation=-1):
        super(GRADUATE, self).__init__()
        
        dim_stft = max_freq
        self.rep_KV = rep_KV
        self.recover_type = recover_type
        self.dualDomain_type = dualDomain_type
        self.ablation = ablation
        self.wavelength = wavelength

        # =========================================== #
        #             Time domain branch              #
        # =========================================== #        
        if ablation != 'time':
            self.conformer = Conformer(num_classes=conformer_class, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers)

        # =========================================== #
        #        Time-frequency domain branch         #
        # =========================================== #
        self.stft_recovertype = stft_recovertype
        
        if ablation != 'time-frequency':
            if stft_recovertype == 'crossattn':
                self.stft_posEmb = PositionalEncoding(dim_stft, max_len=wavelength//4-1, return_vec=True)
                self.stft_pos_emb = cross_attn_layer(nhead, dim_stft//nhead, dim_stft//nhead, dim_stft, conformer_class, d_ffn)
            elif stft_recovertype == 'conv':
                self.stft_conv = nn.Sequential(nn.Upsample(scale_factor=1.5),
                                                nn.Conv1d(dim_stft, 16, kernel_size=5, padding='same'),
                                                nn.ReLU(),
                                                nn.Upsample(scale_factor=1.5),
                                                nn.Conv1d(16, 24, kernel_size=5, padding='same'),
                                                nn.ReLU(),
                                                nn.Upsample(wavelength//4-1),
                                                nn.Conv1d(24, 32, kernel_size=7, padding='same'),
                                                nn.ReLU())
                self.stft_proj = nn.Sequential(nn.Linear(32, conformer_class), 
                                                nn.ReLU())

        # =========================================== #
        #             Combine dual-domain             #
        # =========================================== #
        if ablation != 'time' and ablation != 'time-frequency':
            if dualDomain_type == 'crossattn':
                self.crossattn = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class*2, d_ffn)

        # =========================================== #
        #              Restoring module               #
        # =========================================== #
        if recover_type == 'crossattn':
            self.stft_rep_concat_posEmb = PositionalEncoding(conformer_class*2, max_len=wavelength, return_vec=True)            
            self.restoring = cross_attn_layer(nhead, conformer_class*2//nhead, conformer_class*2//nhead, conformer_class*2, conformer_class, d_ffn)
        elif recover_type == 'conv':
            if ablation != 'time' and ablation != 'time-frequency':
                self.recover_conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                                    nn.Conv1d(conformer_class*2, conformer_class*3, kernel_size=5, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Upsample(scale_factor=2),
                                                    nn.Conv1d(conformer_class*3, conformer_class*4, kernel_size=7, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Upsample(wavelength),
                                                    nn.Conv1d(conformer_class*4, conformer_class, kernel_size=7, padding='same'),
                                                    nn.ReLU())
            else:
                self.recover_conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                                    nn.Conv1d(conformer_class, conformer_class*2, kernel_size=5, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Upsample(scale_factor=2),
                                                    nn.Conv1d(conformer_class*2, conformer_class*3, kernel_size=7, padding='same'),
                                                    nn.ReLU(),
                                                    nn.Upsample(wavelength),
                                                    nn.Conv1d(conformer_class*3, conformer_class, kernel_size=7, padding='same'),
                                                    nn.ReLU())

        # =========================================== #
        #                   Decoder                   #
        # =========================================== #    
        self.decoder = nn.ModuleList([cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
                                        for _ in range(dec_layers)]
                                        )

        # =========================================== #
        #                    Output                   #
        # =========================================== #
        self.label_type = label_type
        self.sigmoid = nn.Sigmoid()
        
        if label_type == 'p':
            self.output = nn.Linear(conformer_class, 1)
            self.output_actfn = nn.Sigmoid()
        elif label_type == 'other':
            self.output = nn.Linear(conformer_class, 2)
            self.output_actfn = nn.Softmax(dim=-1)
        elif label_type == 'all':
            self.output = nn.ModuleList([nn.Linear(conformer_class, 1) for _ in range(3)])
            self.output_actfn = nn.Sigmoid()
        
    def forward(self, wave, stft):
        # wave: (batch, 3000, 12)
        wave = wave.permute(0,2,1)
        if self.ablation != 'time':
            out, _ = self.conformer(wave, self.wavelength)
        if self.ablation != 'time-frequency':
            if self.stft_recovertype == 'crossattn':
                stft_posEmb = self.stft_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                stft_out = self.stft_pos_emb(stft_posEmb, stft, stft)
            elif self.stft_recovertype == 'conv':
                stft_out = self.stft_proj(self.stft_conv(stft.permute(0,2,1)).permute(0,2,1))

        # concat encoded representation with stft
        if self.ablation != 'time' and self.ablation != 'time-frequency':
            if self.dualDomain_type == 'concat':
                concat_rep = torch.cat((out, stft_out), dim=-1)
            elif self.dualDomain_type == 'crossattn':
                concat_rep = self.Combicrossattnne(stft_out, out, out)

        if self.recover_type == 'crossattn':
            stft_rep_concat_posEmb = self.stft_rep_concat_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            crossattn_out = self.recover_conv(stft_rep_concat_posEmb, concat_rep, concat_rep)
        elif self.recover_type == 'conv':
            if self.ablation == 'time':
                concat_rep = stft_out
            elif self.ablation == 'time-frequency':
                concat_rep = out

            crossattn_out = self.recover_conv(concat_rep.permute(0,2,1)).permute(0,2,1)

        # decoder
        for i, layer in enumerate(self.decoder):
            if i == 0:
                if not self.rep_KV or self.ablation == 'time':
                    dec_out = layer(crossattn_out, crossattn_out, crossattn_out)
                else:
                    dec_out = layer(crossattn_out, out, out)
            else:
                if not self.rep_KV or self.ablation == 'time':
                    dec_out = layer(dec_out, dec_out, dec_out)
                else:
                    dec_out = layer(dec_out, out, out)
        
        # output layer
        if self.label_type == 'p' or self.label_type == 'other':
            out = self.output_actfn(self.output(dec_out))
        elif self.label_type == 'all':
            out = []
            # 0: detection, 1: P-pahse, 2: S-phase
            for layer in self.output:
                out.append(self.output_actfn(layer(dec_out)))
      
        if self.ablation == 'time':
            seg_out = 0
        # print("out",out.shape)
        return out

def stalta(wf, short_window, long_window, threshold_lambda):
    res = []
    pred_trigger = []
    out_prob = []

    for w in wf:
        out = classic_sta_lta(w[0], short_window, long_window)
        trigger = trigger_onset(out, threshold_lambda, 1)
       
        out_prob.append(out)

        if len(trigger) > 0:
            res.append(True)
        else:
            res.append(False)
            pred_trigger.append(0)
            continue
            
        candidate = [p[0] for p in trigger] 
        pred_trigger.append(candidate[np.argmax(out[candidate])])

    return res, pred_trigger, torch.FloatTensor(out_prob)

# for Ensemble picking
class PositionEmbedding(nn.Module):
    def __init__(self, emb_dim, min_lat, max_lat, min_lon, max_lon):
        super(PositionEmbedding, self).__init__()
        self.emb_dim = emb_dim

        lat_dim = self.emb_dim // 5
        lon_dim = self.emb_dim // 5

        self.lat_coeff = 2 * np.pi * 1. / min_lat * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        self.lon_coeff = 2 * np.pi * 1. / min_lon * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))

        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3

        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)

        self.mask = torch.LongTensor(self.mask.astype('int'))

    def forward(self, station_list):
        lat_base = station_list[:, :, 0:1] * self.lat_coeff
        lon_base = station_list[:, :, 1:2] * self.lon_coeff

        output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                    torch.sin(lon_base), torch.cos(lon_base),], dim=-1)
        output = output[:, :, self.mask]

        return output.to(dtype=torch.float32)

class Ensemble(nn.Module):
    def __init__(self, emb_dim, min_lat, max_lat, min_lon, max_lon, d_ffn, pos_emb=True):
        super(Ensemble, self).__init__()

        self.ispos_emb = pos_emb
        if pos_emb:
            self.pos_emb = PositionEmbedding(emb_dim, min_lat, max_lat, min_lon, max_lon)

        prenorm_dim = emb_dim+6 if pos_emb else 8
        self.pre_norm_coord_attn = nn.Sequential(nn.Linear(prenorm_dim, d_ffn),
                        nn.ReLU(),
                        nn.LayerNorm(d_ffn),)
        
        self.coord_attn = nn.MultiheadAttention(embed_dim=d_ffn, num_heads=2, batch_first=True)

        self.predict = nn.Linear(d_ffn, 1)
        self.out_actfn = nn.Sigmoid()

    def forward(self, x):
        '''
        x: from picking message: (lat, lon, ptime, p-weight, Pa, Pv, picker_type)

        Usage:
            station_list = [[[121.89, 23.56, 3.25, 0, 0.4, 0.03, 0], [121.75, 23.4, 3.9, 1, 0.4, 0.03, 1]], [[121.89, 21.56, 2.25, 1, 0.4, 0.03, 2], [120.75, 23.47, 4, 1, 0.4, 0.03, 3]]]
            station_list = torch.FloatTensor(station_list)

            min_lon = torch.min(station_list[:, :, 0]) 
            min_lat = torch.min(station_list[:, :, 1])
            max_lon = torch.max(station_list[:, :, 0]) 
            max_lat = torch.max(station_list[:, :, 1]) 
            
            m = Ensemble(20, min_lat, max_lat, min_lon, max_lon)
            m(station_list)
        ''' 
        if self.ispos_emb:
            coord = x[:, :, :2]

            # coord_posemb: (batch, n_stations, emb_dim)
            coord_posemb = self.pos_emb(coord)
            
            # station vector: (batch, n_stations, emb_dim+4)
            station_vector = torch.cat([coord_posemb, x[:, :, 2:]], dim=-1)
        else:
            station_vector = x
        
        hidden_state = self.pre_norm_coord_attn(station_vector)
        hidden_state, _ = self.coord_attn(hidden_state, hidden_state, hidden_state)
        
        # predict
        out = self.out_actfn(self.predict(hidden_state))
        
        return out

# For Ensemble picking (Machine learning classifier)
def ML_classifier(clf, toPredict):
    '''
    Training
    from sklearn.svm import SVC

    clf = SVC(probability=True)

    # 100 sampels, 8 features for each sample
    X = torch.rand(100, 8)

    # output probability for each sample (binary classification)
    y = torch.randint(0, 2, (100,))

    # fitting 
    clf.fit(X, y)

    # inferencing
    predicted_probabilities = clf.predict_proba(torch.rand(10, 8))
    '''

    res = []

    out_prob = clf.predict_proba(toPredict)
    print(out_prob)
    for o in out_prob:
        pred = np.argmax(o)

        if pred == 0:
            res.append(False)
        else:
            res.append(True)

    return res
    