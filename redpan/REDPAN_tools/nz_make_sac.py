import obspy
import numpy as np
from obspy import read, UTCDateTime

def stream_from_h5(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream

def zero_pad_stream(slice_st, data_length, zero_pad_range,
        max_pad_slices=3, pad_mode=None, pad_chn_consistent=True):
    '''
    Randomly pad the noise waveform with values on all channels

    '''
    zero_pad = np.random.randint(
        zero_pad_range[0], zero_pad_range[1])
    
    if pad_chn_consistent==True:
        max_pad_seq_num = np.random.randint(max_pad_slices)+1
        pad_len = np.random.multinomial(zero_pad, 
            np.ones(max_pad_seq_num)/max_pad_seq_num)
        pad_len = np.vstack([pad_len, pad_len, pad_len])

        _max_idx = data_length - pad_len[0]
        _insert_idx = np.random.randint(_max_idx)

    elif pad_chn_consistent==False:
        max_pad_seq_num = [np.random.randint(max_pad_slices)+1
            for c in range(3)]
        pad_len = [np.random.multinomial(zero_pad, 
            np.ones(max_pad_seq_num[c])/max_pad_seq_num[c])
            for c in range(3)]
    
    if not pad_mode:
        pad_mode = np.random.permutation([
            'zeros', 'maximum', 'minimum', 'random'
        ])[0]

    # pad the waveform across all channels


    for ch in [0, 1, 2]:
        if pad_chn_consistent==False:
            max_idx = data_length - pad_len[ch]
            insert_idx = np.random.randint(max_idx)
        elif pad_chn_consistent==True:
            insert_idx = _insert_idx

        insert_end_idx = insert_idx + pad_len[ch]

        max_v = [1.5*np.max(slice_st[ch].data) for ch in range(3)]
        min_v = [1.5*np.min(slice_st[ch].data) for ch in range(3)]

        for j in range(len(insert_end_idx)):
            if insert_end_idx[j] >= zero_pad_range[1]:
                insert_end_idx[j] = zero_pad_range[1]
            if pad_mode == 'zeros':
                slice_st[ch].data[
                    insert_idx[j]:insert_end_idx[j]] = 0
            elif pad_mode == 'maximum':
                slice_st[ch].data[
                    insert_idx[j]:insert_end_idx[j]] = max_v[ch]
            elif pad_mode == 'minimum':
                slice_st[ch].data[
                    insert_idx[j]:insert_end_idx[j]] = min_v[ch]
            elif pad_mode == 'random':
                slice_st[ch].data[
                    insert_idx[j]:insert_end_idx[j]] = \
                        np.random.uniform(min_v[0], max_v[0], 
                            insert_end_idx[j]-insert_idx[j])       
    # pad the waveform inconsistently across the channels

    return slice_st

def drop_channel(slice_st, data_length, drop_chn=[0, 1]):
    for s in range(len(slice_st)):
        if s in drop_chn:
            slice_st[s].data = np.zeros(data_length)
    return slice_st