import numpy as np
import torch
import scipy.signal
import traceback
import time
from scipy import interpolate
import matplotlib.pyplot as plt

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

def z_score(wave, pad_option, pad_presignal_length):
    # padding zeros with neighborhood'd values
    # print(torch.any(wave==0))
    # start = time.time()
    # wave, nonzero_flag = padding(wave, pad_option, pad_presignal_length) 
    # print("padding: ", time.time()-start)
    nonzero_flag = [True for _ in range(wave.shape[0])]

    eps = 1e-10

    wave = wave - torch.mean(wave, dim=-1, keepdims=True)
    wave = wave / (torch.std(wave, dim=-1, keepdims=True) + eps) 

    return wave, nonzero_flag

def calc_feats(waveforms):
    CharFuncFilt = 3
    rawDataFilt = 0.939
    small_float = 1.0e-10
    STA_W = 0.6
    LTA_W = 0.015

    # Calculate data using vectorized operations
    data = torch.cumsum(waveforms + small_float, dim=2) * rawDataFilt
    data -= torch.cat((torch.zeros((waveforms.shape[0], waveforms.shape[1], 1)), data[:, :, :-1]), dim=2)

    # Calculate wave_square and diff_square using vectorized operations
    wave_square = torch.square(data)
    diff = torch.cat((data[:, :, 0:1], data[:, :, 1:] - data[:, :, :-1]), dim=2)
    diff_square = torch.square(diff)

    # Calculate wave_characteristic using vectorized operations
    wave_characteristic = wave_square + diff_square * CharFuncFilt

    # Calculate wave_sta and wave_lta using vectorized operations
    sta = torch.zeros((waveforms.shape[0], waveforms.shape[1], 1))
    lta = torch.zeros((waveforms.shape[0], waveforms.shape[1], 1))

    wave_sta = torch.cumsum((waveforms - sta) * STA_W, dim=2)
    wave_lta = torch.cumsum((waveforms - lta) * LTA_W, dim=2)

    # Concatenate 12-dim vector as output
    # print(waveforms.shape, wave_characteristic.shape, wave_sta.shape, wave_lta.shape)
    waveforms = torch.cat((waveforms, wave_characteristic, wave_sta, wave_lta), dim=1)

    return waveforms

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def padding_with_case(wave, pad_case, pad_idx, pad_option, n_neighbor=100):
    wavelength = wave.shape[-1]
    
    if pad_case == 1:
        toPad_value = wave[pad_idx[-1]:pad_idx[-1]+n_neighbor] if pad_idx[-1]+n_neighbor < wavelength else wave[pad_idx[-1]:]
        n_neighbor = toPad_value.shape[0]
        toPad_length = pad_idx[-1] - pad_idx[0]
        
        if n_neighbor == 0:
            return torch.from_numpy(wave)

        if pad_option == 'repeat':
            if not toPad_length <= 1:
                toPad = np.tile(toPad_value, toPad_length // n_neighbor)
                remaining = toPad_length % n_neighbor

                if remaining > 0:
                    toPad = np.concatenate((toPad, toPad_value[:remaining]))
        elif pad_option == 'gaussian':
            toPad = np.random.normal(0, np.random.uniform(0.05, 0.3)*max(toPad_Value), toPad_length)
        elif pad_option == 'mean':
            toPad = np.ones(toPad_length) * np.mean(toPad_value)

        wave[pad_idx[0]:pad_idx[-1]] = toPad

    elif pad_case == 2:
        toPad_value = wave[pad_idx[0]-n_neighbor:pad_idx[0]] if pad_idx[0]-n_neighbor >= 0 else wave[:pad_idx[0]]
        n_neighbor = toPad_value.shape[0]        
        toPad_length = pad_idx[-1] - pad_idx[0]

        if n_neighbor == 0:
            return torch.from_numpy(wave)

        if pad_option == 'repeat':
            if n_neighbor == 0:
                return torch.from_numpy(wave)
            
            if not toPad_length <= 1:
                toPad = np.tile(toPad_value, toPad_length // n_neighbor)
                remaining = toPad_length % n_neighbor
                if remaining > 0:
                    toPad = np.concatenate((toPad, toPad_value[:remaining]))
    
        elif pad_option == 'gaussian':
            toPad = np.random.normal(0, np.random.uniform(0.05, 0.3)*max(toPad_Value), toPad_length)
        elif pad_option == 'mean':
            toPad = np.ones(toPad_length) * np.mean(toPad_value)

        wave[pad_idx[0]:pad_idx[-1]] = toPad

    else:
        toPad_value = wave[pad_idx[0]-n_neighbor:pad_idx[0]] if pad_idx[0]-n_neighbor >= 0 else wave[:pad_idx[0]]
        # toPad_value = wave[:n_neighbor]
        n_neighbor = toPad_value.shape[0]
        toPad_length = pad_idx[-1] - pad_idx[0]

        if n_neighbor == 0:
            return torch.from_numpy(wave)

        if pad_option == 'repeat':
            if n_neighbor == 0:
                return torch.from_numpy(wave)
            
            if not toPad_length <= 1:
                toPad = np.tile(toPad_value, toPad_length // n_neighbor)
                remaining = toPad_length % n_neighbor
                
                if remaining > 0:
                    toPad = np.concatenate((toPad, toPad_value[:remaining]))

        elif pad_option == 'gaussian':
            toPad = np.random.normal(0, np.random.uniform(0.05, 0.3)*max(toPad_Value), toPad_length)
        elif pad_option == 'mean':
            toPad = np.ones(toPad_length) * np.mean(toPad_value)

        # wave = np.concatenate((toPad, wave[:pad_idx[0]]))
        wave[pad_idx[0]:pad_idx[-1]] = toPad

    return torch.from_numpy(wave)

# def padding(a, pad_option, pad_presignal_length):
#     '''
#     Three types of padding:
#     1) Zeros is at the beginning of the waveform.
#     2) Zeros is at the end of the waveform.
#     3) Zeros is at the middle of the waveform.
#     Note that multiple cases may occur in a single waveform.
#     '''
#     nonzero_flag = [True for _ in range(a.shape[0])]
#     for batch in range(a.shape[0]):
#         try:
#             wave = a[batch]
#             backup_wave = a[batch].clone()

#             # check waveform full of zeros
#             if torch.all(wave == 0):
#                 nonzero_flag[batch] = False
#                 continue

#             # finding zero values alone Z, N, E axis
#             zeros = [[zero_runs(wave[i].numpy())] for i in range(wave.shape[0])]
       
#             # padding follows the order: Z -> N -> E
#             batch_pad = False
#             for i in range(len(zeros)):
#                 # There is no zero in the trace
#                 if zeros[i][0].shape[0] == 0:
#                     continue

#                 for row, j in enumerate(zeros[i][0]):
#                     isPad = False

#                     # check first row contain "0" or not, if not, then padding_case 1 is not satisfied.
#                     if j[0] == 0:
#                         # padding case 1
#                         wave[i] = padding_with_case(wave[i].numpy(), 1, j, pad_option, pad_presignal_length)
#                         isPad = True
#                         batch_pad = True

#                     # check the last row contain "wavelength-1" or not, if not, then padding_case 3 is not satisfied.
#                     if j[-1] == wave.shape[-1]:
#                         # padding case 3
#                         wave[i] = padding_with_case(wave[i].numpy(), 3, j, pad_option, pad_presignal_length)
#                         isPad = True
#                         batch_pad = True

#                     # check the middle rows
#                     if not isPad:
#                         wave[i] = padding_with_case(wave[i].numpy(), 2, j, pad_option, pad_presignal_length)
#                         batch_pad = True

#                 a[batch] = wave

#             if batch_pad:
#                 nonzero_flag[batch] = False
#         except Exception as e:
#             # print(e)
#             a[batch] = backup_wave
            
#     return a, nonzero_flag

def padding(a, pad_option, pad_presignal_length):
    nonzero_flag = [False for _ in range(a.shape[0])]

    for batch in range(a.shape[0]):
        backup_wave = a[batch].clone()

        for j in range(a.shape[1]):    
            try:
                if torch.all(a[batch, j] != 0):
                    nonzero_flag[batch] = True
                    continue
                b = fill(a[batch, j])
                
                a[batch, j] = b
            except Exception as e:
                # print(e)
                # print(traceback.format_exc())
                a[batch] = backup_wave

    return a, nonzero_flag

def fill(trace):
    trace = trace.cpu().numpy()
    zeros, x = (trace == 0), lambda z: z.nonzero()[0]

    trace[zeros] = interpolate.interp1d(x(~zeros), trace[~zeros], kind='linear', fill_value="extrapolate")(x(zeros))
    
    scale = np.random.uniform(0, 0.3) * np.mean(trace[zeros])
    noise = np.random.randn(trace[zeros].shape[0]).astype(trace[zeros].dtype) * scale
    
    trace[zeros] = trace[zeros] + noise
    return torch.from_numpy(trace)

def STFT(wave):
    acc = wave[:, 0]**2 + wave[:, 1]**2 + wave[:, 2]**2
    acc = np.sqrt(acc)
    f, t, Zxx = scipy.signal.stft(acc, nperseg=20, nfft=64, axis=-1)
    real = np.abs(Zxx.real)
    
    real = real[:, :12]

    return torch.FloatTensor(np.transpose(real, (0, 2, 1)))
