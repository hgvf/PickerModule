import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from obspy import read
import sys
from REDPAN_tools.data_utils import sac_len_complement
from REDPAN_tools.data_utils import stream_standardize
from REDPAN_tools.mtan_ARRU import unets
from REDPAN_tools.data_utils import PhasePicker
from REDPAN_tools.REDPAN_picker import extract_picks
from REDPAN_tools.data_utils import picker_info
from scipy.signal import find_peaks
from tqdm import tqdm
import pickle
# from REDPAN_tools.mtan_ARRU_light import unets
from scipy import stats

def find_nearest_P(gt, pred):
    min_diff = 10000
    res = 0

    for p in pred:
        if np.abs(p-gt) < min_diff:
            min_diff = np.abs(p-gt)
            res = p

    return res

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
gpu_devices = tf.config.experimental.list_physical_devices('CPU')

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s : %(asctime)s : %(message)s')

## (1) Load model
pred_npts = 3000 # model input length

save_path = "pretrain_REDPAN_30s_STEAD"
# model_h5 = '/mnt/nas5/weiwei/RED-PAN/pretrained_model/REDPAN_30s/train.hdf5'
model_h5 = "/mnt/disk1/weiwei/MQTT/redpan/train.hdf5"
print('Loading checkpoint: ', model_h5)
frame = unets(input_size=(pred_npts, 3))
model = frame.build_mtan_R2unet(
    model_h5, input_size=(pred_npts, 3)
)  
# print(model.summary())
postprocess_config = {
    "mask_trigger": [0.3, 0.3], # trg_on and trg_off threshold
    "mask_len_thre": 0.5, # minimum length of mask in seconds
    "mask_err_win": 0.5, # potential window in seconds for mask error
    "detection_threshold": 0.5, # detection threshold for mask
    "P_threshold": 0.3, # detection threshold for P
    "S_threshold": 0.3 # detection threshold for S
}
dt = 0.01

# picker = PhasePicker(
#     model=model, 
#     pred_npts=pred_npts,
#     pred_interval_sec=10,
#     dt=dt, 
#     postprocess_config=postprocess_config
# )

pick_args = {
        "detection_threshold": 0.5, 
        "P_threshold": 0.3,
        "S_threshold": 0.3
    }

p_arrival = [750, 1500, 2000, 2500, 2750]
# subset_dir = ['CWBSN', 'TSMIP', 'CWB_noise']
subset_dir = ['STEAD']

Picks, Masks = model.predict(np.random.rand(10, 3000, 3))
print(Picks)
# Start inferencing
for p_time in p_arrival:
    print('P-arrival: ', p_time)
    tp, fp, tn, fn = 0, 0, 0, 0
    diff = []
    for subset in subset_dir:
        cnt = 0
        df = pd.read_csv(f"/mnt/disk4/weiwei/seismic_datasets/Taiwan_redpanFormat/{subset}/evlist.csv")
        df_test = df.loc[df['split'] == 'test']

        dirname = f"/mnt/disk4/weiwei/seismic_datasets/Taiwan_redpanFormat/{subset}/test"
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
            toPad_flag = False
            cnt += 1
            if subset != 'STEAD' and subset != 'INSTANCE':
                trc_name = f"{row['source_id']}.{row['station']}.{row['channel']}.{row['network']}.{row['loc']}*.sac"
            else:
                trc_name = f"{row['source_id']}.{row['station']}.{row['channel']}.{row['network']}.{int(row['loc'])}*.sac" 
            
            d_name = os.path.join(dirname, str(row['source_id']))
            wf_filepath = f"{d_name}/{trc_name}"

            wf = read(wf_filepath)
            wf.sort() # should be E-N-Z order

            gt_P = wf[0].stats['sac']['t1']

            if not np.isnan(gt_P):
                # require padding
                if gt_P < p_time:
                    toPad_length = int(p_time) - int(gt_P)
                    toPad_flag = True

                    start = 0
                    end = 3000 - toPad_length
                else:
                    start = int(gt_P) - p_time
                    # print(f"start: {start}, gt_P: {gt_P}")
                    if start < 0:
                        start = 0
                gt_P = p_time
            else:
                start = 0

            for s in wf:
                if not toPad_flag:
                    s.data = s.data[start:start+3000]
                else:
                    s.data = s.data[start:start+end]

                    toPad_value = s.data[start:start+10]
                    toPad = np.tile(toPad_value, toPad_length // 10)
                    remaining = toPad_length % 10
                    if remaining > 0:
                        toPad = np.concatenate((toPad, toPad_value[:remaining]))
                    s.data = np.hstack((toPad, s.data))

                data_std = np.std(s.data)
                if data_std == 0:
                    data_std = 1
                s.data /= data_std
                s.data[np.isinf(s.data)] = 0
                s.data[np.isnan(s.data)] = 0

            wf = wf.detrend("demean").filter("bandpass", freqmin=1, freqmax=45)
            # print(wf)
            try:
                trc_data = np.array([W.data for W in wf])
                batch_size = 1
                batch_trc = np.repeat(trc_data.T[np.newaxis, ...], batch_size, axis=0)
                # print(batch_trc.shape)
                # P, S, M = picker.predict(batch_trc)
                Picks, Masks = model.predict(batch_trc)
                
                # print(Picks.shape, Masks.shape)
                # plt.subplot(311)
                # plt.plot(batch_trc[0])
                # plt.subplot(312)
                # plt.plot(Picks[0])
                # plt.subplot(313)
                # plt.plot(Masks[0])
                # plt.savefig(f"./tmp/{cnt}.png")
                # plt.clf()

            except Exception as e:
                pass
                # print(e)
                # wf.plot()

            # pred_p = picker_info(M, P, S, pick_args)
            pred_p = find_peaks(Picks[0, :, 0], height=pick_args["P_threshold"], distance=1)[0]
            # print('pred_p: ', pred_p)
            # Ground-truth is not trigger
            if np.isnan(gt_P):
                if len(pred_p) == 0:
                    tn += 1
                    # print('tn')
                else:
                    fp += 1
                    # print('fp')
            else:
                if len(pred_p) == 0:
                    fn += 1
                    # print('fn')
                else:
                    left_edge = gt_P - 50 if gt_P-50 >= 0 else 0
                    right_edge = gt_P + 50 if gt_P+50 < 3000 else 2999
                    
                    pred_p = find_nearest_P(gt_P, pred_p)
                    diff.append(pred_p-gt_P)
                    if pred_p >= left_edge and pred_p <= right_edge:
                        tp += 1
                        # print('tp')
                    else:
                        fp += 1
                        # print('fp')

                    # print(f"Ground-truth-> P-arrival: {gt_P}")
                    # print(f"Prediction-> P-arrival: {pred_p}")
                    # print(f"Result-> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            # plt.subplot(411)
            # plt.plot(wf[0].data)
            # plt.subplot(412)
            # plt.plot(wf[1].data)
            # plt.subplot(413)
            # plt.plot(wf[2].data)
            # plt.subplot(414)
            # plt.plot(Picks[0, :, 0])
            # plt.axvline(x=pred_p, color='r')
            # plt.show()

            # if cnt == 30:
            #     break
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
    fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
    with open(f'./results/{save_path}.log', 'a') as f:
        f.write(f"P-arrival: {p_time}\n")
        f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"F1-score: {fscore}\n")
        f.write('trigger_mean=%.4f, trigger_std=%.4f\n' %(np.mean(diff)/100, np.std(diff)/100))
        f.write('='*50)
        f.write('\n')

    with open(f'./results/{save_path}_{p_time}_diff.pkl', 'wb') as f:
        pickle.dump(diff, f)

