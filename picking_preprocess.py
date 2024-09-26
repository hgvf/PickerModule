import numpy as np
import torch
import scipy.signal
import traceback
import time
import matplotlib.pyplot as plt
from scipy import interpolate
from multiprocessing import Process

def v_to_a(waveform, idx, sampling_rate=100.0):
    waveform = waveform.numpy()
    waveform[idx] = np.gradient(waveform[idx], 1.0/sampling_rate, axis=-1)

    return torch.FloatTensor(waveform)

def filter(waveform, sos):
    device = waveform.device
    waveform = waveform.cpu()
    
    start = time.time()
    # 加長波型，減少因為 filter 導致波型異常現象
    n_append = 100
    n_repeat = 25

    # 用 waveform 前後 100 timesteps 來補足
    tmp_front = waveform[:, :, :n_append].repeat(1, 1, n_repeat)
    tmp_end = waveform[:, :, -n_append:].repeat(1, 1, n_repeat)

    # tmp = torch.zeros(waveform.shape[0], waveform.shape[1], n_append)
    toFilter = torch.cat((tmp_front, waveform, tmp_end), dim=-1)
    res = torch.FloatTensor(scipy.signal.sosfilt(sos, toFilter, axis=-1))

    return torch.FloatTensor(res[:, :, n_append*n_repeat:-n_append*n_repeat]).to(device)

def pad_zero(wave):
    device = wave.device
    wave = wave.cpu().numpy()

    zero_points = (wave == 0)
    shifted_data = np.roll(wave, shift=1, axis=2)
    wave[zero_points] = shifted_data[zero_points]

    return torch.FloatTensor(wave).to(device)

def z_score(wave):
    eps = 1e-10

    wave = pad_zero(wave)

    wave = wave - torch.mean(wave, dim=-1, keepdims=True)
    wave = wave / (torch.std(wave, dim=-1, keepdims=True) + eps) 

    return wave

def calc_feats(waveforms):
    CharFuncFilt = 3
    rawDataFilt = 0.939
    small_float = 1.0e-10
    STA_W = 0.6
    LTA_W = 0.015

    # Calculate data using vectorized operations
    data = torch.cumsum(waveforms + small_float, dim=2) * rawDataFilt
    data -= torch.cat((torch.zeros((waveforms.shape[0], waveforms.shape[1], 1)).to(waveforms.device), data[:, :, :-1]), dim=2)

    # Calculate wave_square and diff_square using vectorized operations
    wave_square = torch.square(data)
    diff = torch.cat((data[:, :, 0:1], data[:, :, 1:] - data[:, :, :-1]), dim=2)
    diff_square = torch.square(diff)

    # Calculate wave_characteristic using vectorized operations
    wave_characteristic = wave_square + diff_square * CharFuncFilt

    # Calculate wave_sta and wave_lta using vectorized operations
    sta = torch.zeros((waveforms.shape[0], 3)).to(waveforms.device)
    lta = torch.zeros((waveforms.shape[0], 3)).to(waveforms.device)
    
    wave_sta = torch.zeros((waveforms.shape)).to(waveforms.device)
    wave_lta = torch.zeros((waveforms.shape)).to(waveforms.device)
    
    # Compute esta, the short-term average of edat
    for i in range(waveforms.shape[2]):
        sta += 0.6 * (wave_characteristic[:, :, i] - sta)

        # sta's output vector
        wave_sta[:, :, i] = sta

    # Compute esta, the short-term average of edat
    for i in range(waveforms.shape[2]):
        lta += 0.015 * (wave_characteristic[:, :, i] - lta)

        # lta's output vector
        wave_lta[:, :, i] = lta

    # Concatenate 12-dim vector as output
    # print(waveforms.shape, wave_characteristic.shape, wave_sta.shape, wave_lta.shape)
    waveforms = torch.cat((waveforms, wave_characteristic, wave_sta, wave_lta), dim=1)

    return waveforms

def STFT(wave):
    acc = wave[:, 0]**2 + wave[:, 1]**2 + wave[:, 2]**2
    acc = np.sqrt(acc)
    f, t, Zxx = scipy.signal.stft(acc, nperseg=20, nfft=64, axis=-1)
    real = np.abs(Zxx.real)
    
    real = real[:, :12]

    return torch.FloatTensor(np.transpose(real, (0, 2, 1)))

# For STA/LTA
def characteristic(wf):
    # demean
    wf = wf - torch.mean(wf, dim=-1, keepdims=True)

    # diff
    diff = torch.diff(wf, dim=-1)
    
    results = torch.cat((wf[..., 0].unsqueeze(-1), wf[..., 1:] + diff ** 2), dim=-1)
    
    return results

