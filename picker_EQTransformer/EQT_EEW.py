import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn.parallel import DataParallel

import multiprocessing
from multiprocessing import Process, Manager, Array, Value, Queue, Pool
from queue import Queue
from multiprocessing.managers import BaseManager
from itertools import compress

import ctypes as c
import random
import pandas as pd 
import os
import glob
import bisect
import shutil
import uuid
import datetime

from tqdm import tqdm
import sys
sys.path.append('../')

from ctypes import c_char_p
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone
from collections import deque
from picking_preprocess import *
from picking_utils import *
from Pavd_module import *

import seisbench.models as sbm

# time consuming !
import matplotlib.pyplot as plt
import json
import paho.mqtt.client as mqtt
from obspy import read
import struct

# for shared lass object
class MyManager(BaseManager): pass
def Manager_Pavd():
    m = MyManager()
    m.start()
    return m 

class Mqtt():
    def __init__(self,):
        
        self.cnt = 0
        self.init_shared_params()
        self.station_chunk()

    # 建立連線（接收到 CONNACK）的回呼函數
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code " + str(rc))
            client.subscribe("RSD24bits/#")
        else:
            print("Failed to connect, ", rc)
            client.disconnect()

    # 接收訊息（接收到 PUBLISH）的回呼函數
    def on_message(self, client, userdata, msg):
        # 0.0003 s on average

        # parse the package
        msg = msg.payload
        network, station, location, channel, nsamp, samprate, starttime, endtime = struct.unpack(f"<2s5s2s3sIddd", msg[0:40])
        network = network.decode().strip()
        station = station.decode().strip()
        location = location.decode().strip()
        channel = channel.decode().strip()
        scnl = f"{station}_{channel}_{network}_{location}"
        data = np.array(struct.unpack(f"<{nsamp:d}i", msg[40:]), dtype=np.int32)
        
        # check the station is in the partial station list
        # if int(self.env_config["CHUNK"]) != -1:
        #     if station not in self.partial_station_list:
        #         return 

        # update the key_index
        if scnl not in self.key_index:
            if scnl not in self.key_index:
                self.key_index[scnl] = int(self.key_cnt.value)
                self.key_cnt.value += 1

        # append the data in package into shared waveform buffer
        startIndex = int(starttime*self.env_config['SAMP_RATE']) - int(self.waveform_buffer_start_time.value)
            
        data = data.copy().astype(np.float32)
        self.waveform_buffer[self.key_index[scnl]][startIndex:startIndex+nsamp] = torch.from_numpy(data)
        
        # send information to module for calculating Pa, Pv, and Pd
        if channel[-1] == 'Z':
            msg_to_pavd = {scnl: data}
            self.pavd_scnl.put(msg_to_pavd)
    
    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print("Unexpected disconnection: ", str(rc))
            # Optionally, add reconnection logic here
            client.reconnect()
    
    def on_log(self, client, userdata, level, buf):
        # print("log: ", buf)
        pass

    def activate_mqtt(self):
        # 建立 MQTT Client 物件
        self.client = mqtt.Client()
        # 設定建立連線回呼函數 (callback function)
        self.client.on_connect = self.on_connect

        # 設定接收訊息回呼函數
        self.client.on_message = self.on_message

        self.client.on_disconnect = self.on_disconnect    
        self.client.on_log = self.on_log

        # 連線至 MQTT 伺服器（伺服器位址,連接埠）, timeout=6000s
        self.client.connect("140.118.127.89", 1883, 6000)
        print("connect!")
        
        # 進入無窮處理迴圈
        # self.client.loop_forever()
        self.client.loop_start()

    def start(self):
        try:
            self.activate_mqtt()

            time_mover = Process(target=TimeMover, args=(self.waveform_buffer, self.env_config, self.nowtime, self.waveform_buffer_start_time))
            time_mover.start()

            pavd_sender = Process(target=PavdModule_sender, args=(self.env_config['CHECKPOINT_TYPE'], self.pavd_calc, self.waveform_comein, self.waveform_comein_length, self.pavd_scnl, self.waveform_scnl))
            pavd_sender.start()

            picker = Process(target=Picker, args=(self.waveform_buffer, self.key_index, self.nowtime, self.waveform_buffer_start_time, self.env_config, self.key_cnt, self.stationInfo, self.device,
                                                    self.waveform_save_picktime, (self.notify_tokens, self.waveform_tokens), self.waveform_save, self.waveform_save_res, self.waveform_save_prediction, 
                                                    self.waveform_save_TF, self.save_info, self.waveform_save_waveform_starttime, self.logfilename_pick, self.logfilename_original_pick, self.logfilename_notify, self.logfilename_cwb_pick, self.upload_TF, 
                                                    self.restart_cond, self.keep_wave_cnt, self.remove_daily, self.waveform_plot_TF, self.plot_info, self.waveform_plot_wf, self.waveform_plot_out, self.waveform_plot_picktime, self.plot_time,
                                                    self.notify_TF, self.toNotify_pickedCoord, self.n_notify, self.picked_waveform_save_TF, self.scnl, self.avg_pickingtime, self.median_pickingtime, self.n_pick, self.logfilename_stat, self.pick_stat_notify,
                                                    self.pavd_sta))
            picker.start()

            pavd_processes = []
            for i in range(7):
                pavd_calculator = Process(target=Pavd_calculator, args=(self.pavd_calc[i], self.waveform_comein[i], self.waveform_scnl[i], self.waveform_comein_length[i], self.pavd_sta, self.stationInfo, self.env_config))
                pavd_calculator.start()
                pavd_processes.append(pavd_calculator)

            notifier = Process(target=Notifier, args=(self.notify_TF, self.toNotify_pickedCoord, self.notify_tokens, self.n_notify, int(self.env_config['CHUNK']), self.env_config['CHECKPOINT_TYPE']))
            notifier.start()

            wave_shower = Process(target=Shower, args=(self.waveform_plot_TF, self.plot_info, self.waveform_plot_wf, self.waveform_plot_out, self.waveform_plot_picktime, self.waveform_tokens, self.env_config['CHECKPOINT_TYPE']))
            wave_shower.start()

            uploader = Process(target=Uploader, args=(self.logfilename_pick, self.logfilename_notify, self.logfilename_original_pick, self.logfilename_cwb_pick, self.logfilename_stat, self.env_config['TRC_PATH'], self.upload_TF, self.avg_pickingtime, self.median_pickingtime, self.n_pick, self.cwb_avg_pickingtime, self.cwb_median_pickingtime, self.cwb_n_pick, self.calc_cwbstat, self.env_config['CHECKPOINT_TYPE'], self.pick_stat_notify))
            uploader.start()            

            remover = Process(target=Remover, args=(self.remove_daily, self.env_config['WAVE_EXPIRED'], self.env_config['CHECKPOINT_TYPE']))
            remover.start()

            time_mover.join()
            pavd_sender.join()
            picker.join()
            notifier.join()
            wave_shower.join()
            uploader.join()
            remover.join()

            for w in pavd_processes:
                w.join()
            
        except KeyboardInterrupt:
            for w in wavesaver:
                w.terminate()
                w.join()

    def init_shared_params(self):
        self.n_buffer = 1

        # create multiprocessing manager to maintain the shared variables
        manager = Manager()
        self.queue = [manager.Queue(maxsize=50000) for _ in range(self.n_buffer)]
        self.env_config = manager.dict()
        for k, v in dotenv_values(".env").items():
            self.env_config[k] = v

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        # get the candidate line notify tokens
        self.notify_tokens, self.waveform_tokens = load_tokens(self.env_config['NOTIFY_TOKENS'], self.env_config['WAVEFORM_TOKENS'])
        
        # get the station's info
        if self.env_config['SOURCE'] == 'Palert':
            self.stationInfo = get_PalertStationInfo(self.env_config['PALERT_FILEPATH'])
        elif self.env_config['SOURCE'] == 'CWB':
            self.stationInfo = get_CWBStationInfo(self.env_config['STAEEW_FILEPATH'])
            # stationInfo = get_StationInfo(self.env_config['NSTA_FILEPATH'], (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))
        elif self.env_config['SOURCE'] == 'TSMIP':
            self.stationInfo = get_TSMIPStationInfo(self.env_config['TSMIP_FILEPATH'])
        else:
            self.stationInfo = get_StationInfo(self.env_config['NSTA_FILEPATH'], (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))

        # restart the system in process
        self.restart_cond = Value('d', int(0))

        if int(self.env_config['CHUNK']) == -1:
            n_stations = len(self.stationInfo)
        else:
            n_stations = int(self.env_config["N_PREDICTION_STATION"])

        # a deque from time-3000 to time for time index
        self.nowtime = Value('d', int(time.time()*100))
        self.waveform_buffer_start_time = Value('d', self.nowtime.value-3000)

        # a counter for accumulating key's count
        self.key_cnt = Value('d', int(0))

        # a dict for checking scnl's index of waveform
        self.key_index = manager.dict()
      
        self.waveform_buffer = torch.zeros((n_stations*6, int(self.env_config["STORE_LENGTH"]))).share_memory_()

        # parameters for WaveKeeper
        self.waveform_save = torch.empty((n_stations, 3, int(self.env_config['PREDICT_LENGTH']))).share_memory_()
        self.waveform_save_prediction = torch.empty((n_stations, int(self.env_config['PREDICT_LENGTH']))).share_memory_()
        self.waveform_save_res = torch.empty((n_stations,)).share_memory_()
        self.waveform_save_picktime = torch.empty((n_stations,)).share_memory_()
        self.waveform_save_TF = Value('d', int(0))
        self.waveform_plot_TF = Value('d', int(0))
        self.waveform_save_waveform_starttime = manager.Value(c_char_p, 'Hello')
        self.plot_time = manager.Value(c_char_p, 'Hello')
        self.save_info = manager.dict()

        # parameters for Shower
        self.plot_info = manager.Value(c_char_p, 'hello')
        self.waveform_plot_wf = torch.empty((1, 3, int(self.env_config['PREDICT_LENGTH']))).share_memory_()
        self.waveform_plot_out = torch.empty((1, int(self.env_config['PREDICT_LENGTH']))).share_memory_()
        self.waveform_plot_picktime = Value('d', int(0))

        # parameters for uploader
        self.logfilename_pick = manager.Value(c_char_p, 'hello')
        self.logfilename_notify = manager.Value(c_char_p, 'hello')
        self.logfilename_original_pick = manager.Value(c_char_p, 'hello')
        self.logfilename_cwb_pick = manager.Value(c_char_p, 'hello')
        self.logfilename_stat = manager.Value(c_char_p, 'hello')
        self.upload_TF = Value('d', int(0))
        self.avg_pickingtime = Value('d', int(0))
        self.median_pickingtime = Value('d', int(0))
        self.keep_wave_cnt = Value('d', int(0))
        self.n_pick = Value('d', int(0))
        self.pick_stat_notify = Value('d', int(0))
        self.cwb_avg_pickingtime = Value('d', int(0))
        self.cwb_median_pickingtime = Value('d', int(0))
        self.cwb_n_pick = Value('d', int(0))
        self.calc_cwbstat = Value('d', int(0))

        # parameters for notifier
        self.notify_TF = Value('d', int(0))
        self.toNotify_pickedCoord = manager.dict()
        self.n_notify = Value('d', int(0))

        # parameters for remover
        self.remove_daily = Value('d', int(0))

        # parameters for pickwaveform_saver
        self.picked_waveform_save_TF = Value('d', int(0))
        self.scnl = manager.dict()

        # parameters for CWB pickwaveform_saver
        self.cwb_picked_waveform_save_TF = Value('d', int(0))
        self.cwb_scnl = manager.dict()

        # parameters for Pavd_calculator
        self.pavd_sta = manager.dict()

        self.pavd_calc = [Value('d', int(0)) for _ in range(7)]
        self.waveform_comein = [torch.empty((1, 500)).share_memory_() for _ in range(7)]
        self.waveform_comein_length = [Value('d', int(0)) for _ in range(7)]
        self.waveform_scnl = [manager.Value(c_char_p, 'hello') for _ in range(7)]
        self.pavd_scnl = manager.Queue()

    def station_chunk(self):
        self.partial_station_list = None
        if int(self.env_config["CHUNK"]) != -1:
            # 如果是 TSMIP，額外再做分區，並把分區結果存下來
            if self.env_config['SOURCE'] == 'TSMIP' and int(self.env_config['CHUNK']) != -1:
                self.partial_station_list = ForTSMIP_station_selection(self.stationInfo, int(self.env_config["N_PREDICTION_STATION"]))

            self.partial_station_list, _ = station_selection(sel_chunk=int(self.env_config["CHUNK"]), station_list=self.stationInfo, opt=self.env_config['SOURCE'], build_table=False, n_stations=int(self.env_config["N_PREDICTION_STATION"]))

def TimeMover(waveform_buffer, env_config, nowtime, waveform_buffer_start_time):
    print('Starting TimeMover...')

    while True:
        try:
            # move the time window of timeIndex and waveform every 5 seconds
            if int(time.time()*100) - nowtime.value >= 500:
                waveform_buffer[:, 0:int(env_config["STORE_LENGTH"])-500] = waveform_buffer[:, 500:int(env_config["STORE_LENGTH"])]
                
                # the updated waveform is fill in with 0
                waveform_buffer[:, int(env_config["STORE_LENGTH"])-500:int(env_config["STORE_LENGTH"])] = torch.zeros((waveform_buffer.shape[0],500))
                waveform_buffer_start_time.value += 500
                nowtime.value += 500
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (TimeMover): {e}\n")
                pif.write(f"Trace back (TimeMover): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()

            continue

# picking: pick and send pick_msg to PICK_RING
def Picker(waveform_buffer, key_index, nowtime, waveform_buffer_start_time, env_config, key_cnt, stationInfo, device,
            waveform_save_picktime, tokens, waveform_save, waveform_save_res, waveform_save_prediction, waveform_save_TF, save_info, waveform_save_waveform_starttime, 
            logfilename_pick, logfilename_original_pick, logfilename_notify, logfilename_cwb_pick, upload_TF, restart_cond, keep_wave_cnt, remove_daily,
            waveform_plot_TF, plot_info, waveform_plot_wf, waveform_plot_out, waveform_plot_picktime, plot_time,
            notify_TF, toNotify_pickedCoord, n_notify, picked_waveform_save_TF, scnl, avg_pickingtime, median_pickingtime, n_pick, logfilename_stat, pick_stat_notify,
            pavd_sta):
    
    print('Starting Picker...')
    
    # conformer picker
    model_path = env_config["PICKER_CHECKPOINT_PATH"]
    if env_config["CHECKPOINT_TYPE"] == 'GRADUATE':
        in_feat = 12
        model = GRADUATE(conformer_class=8, d_ffn=128, nhead=4, enc_layers=2, dec_layers=1, d_model=12, wavelength=int(env_config['PREDICT_LENGTH'])).to(device)
        torch.cuda.empty_cache()
    elif env_config["CHECKPOINT_TYPE"] == 'eqt':
        in_feat = 3
        model = sbm.EQTransformer(in_samples=int(env_config['PREDICT_LENGTH'])).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    channel_tail = ['Z', 'N', 'E']

    # butterworth filter
    N=5
    Wn=[1, 10]
    btype='bandpass'
    analog=False
    sampling_rate=100.0
    _filt_args = (N, Wn, btype, analog)
    sos = scipy.signal.butter(*_filt_args, output="sos", fs=sampling_rate)

    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day
    system_hour = cur.hour
    
    # Neighbor_table: 蒐集每個測站方圓 X km 內所有測站的代碼
    _, neighbor_table = station_selection(sel_chunk=int(env_config["CHUNK"]), station_list=stationInfo, opt=env_config['SOURCE'], build_table=True, n_stations=int(env_config["N_PREDICTION_STATION"]), threshold_km=float(env_config['THRESHOLD_KM']),
                                            nearest_station=int(env_config['NEAREST_STATION']), option=env_config['TABLE_OPTION'])

    # sleep 120 seconds, 讓波型先充滿 noise，而不是 0
    print('pending...')
    for _ in tqdm(range(int(env_config['SLEEP_SECOND']))):
        time.sleep(1)

    # handle line notify tokens
    line_tokens, waveform_tokens = tokens
    line_token_number, wave_token_number = 0, 0

    # use for filter the picked stations that is picked before
    pick_record = {}

    # 統計每日 picking 數量與時間點
    pick_stat = []

    toPredict_idx, toPredict_scnl = [], []
    prev_key_index = {}

    while True:
        try:
            cur = datetime.fromtimestamp(time.time())
            system_record_time = time.time()

            # 每小時發一個 notify，證明系統還活著
            if f"{system_year}-{system_month}-{system_day}-{system_hour}" != f"{cur.year}-{cur.month}-{cur.day}-{cur.hour}":
                wave_token_number = random.sample(range(len(waveform_tokens)), k=1)[0]
                wave_token_number = alive_notify(waveform_tokens, wave_token_number)
                system_hour = cur.hour

            # 已經是系統時間的隔天，檢查有沒有過舊的 log file，有的話將其刪除
            if f"{system_year}-{system_month}-{system_day}" != f"{cur.year}-{cur.month}-{cur.day}":
                toDelete_picking = cur - timedelta(days=int(env_config['DELETE_PICKINGLOG_DAY']))
                toDelete_notify = cur - timedelta(days=int(env_config['DELETE_NOTIFYLOG_DAY']))

                toDelete_picking_filename = f"./log/picking/{env_config['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_picking_chunk{env_config['CHUNK']}.log"
                toDelete_original_picking_filename = f"./log/original_picking/{env_config['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_original_picking_chunk{env_config['CHUNK']}.log"
                toDelete_notify_filename = f"./log/notify/{env_config['CHECKPOINT_TYPE']}_{toDelete_notify.year}-{toDelete_notify.month}-{toDelete_notify.day}_notify_chunk{env_config['CHUNK']}.log"
                toDelete_cwbpicker_filename = f"./log/CWBPicker/{env_config['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_picking.log"
                toDelete_exception_filename = f"./log/exception/{env_config['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}.log"

                if os.path.exists(toDelete_picking_filename):
                    os.remove(toDelete_picking_filename)
                if os.path.exists(toDelete_original_picking_filename):
                    os.remove(toDelete_original_picking_filename)
                if os.path.exists(toDelete_notify_filename):
                    os.remove(toDelete_notify_filename)
                if os.path.exists(toDelete_cwbpicker_filename):
                    os.remove(toDelete_cwbpicker_filename)
                if os.path.exists(toDelete_exception_filename):
                    os.remove(toDelete_exception_filename)

                # upload files
                logfilename_pick.value = f"./log/picking/{env_config['CHECKPOINT_TYPE']}_{system_year}-{system_month}-{system_day}_picking_chunk{env_config['CHUNK']}.log"
                logfilename_original_pick.value = f"./log/original_picking/{env_config['CHECKPOINT_TYPE']}_{system_year}-{system_month}-{system_day}_original_picking_chunk{env_config['CHUNK']}.log"
                logfilename_notify.value = f"./log/notify/{env_config['CHECKPOINT_TYPE']}_{system_year}-{system_month}-{system_day}_notify_chunk{env_config['CHUNK']}.log"
                logfilename_cwb_pick.value = f"./log/CWBPicker/{env_config['CHECKPOINT_TYPE']}_{system_year}-{system_month}-{system_day}_picking.log"
                logfilename_stat.value = f"./log/statistical/{env_config['CHECKPOINT_TYPE']}_{system_year}-{system_month}-{system_day}_picker_stat.log"

                # 將每日統計結果傳給 Uploader
                if len(pick_stat) > 0:
                    avg_pickingtime.value = round((int(env_config['PREDICT_LENGTH']) - np.mean(pick_stat)) / 100, 2)
                    median_pickingtime.value = round((int(env_config['PREDICT_LENGTH']) - np.median(pick_stat)) / 100, 2)
                    n_pick.value = len(pick_stat)
                else:
                    avg_pickingtime.value = 0
                    median_pickingtime.value = 0
                    n_pick.value = 0

                pick_stat = []

                # 把保留波形的參數歸零
                keep_wave_cnt.value *= 0
                upload_TF.value += 1
                remove_daily.value += 1

                system_year, system_month, system_day = cur.year, cur.month, cur.day

            isPlot = False
            cur_waveform_starttime = datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)
            cur_waveform_buffer, cur_key_index = waveform_buffer.clone(), key_index.copy()

            # skip if there is no waveform in buffer or key_index is collect faster than waveform
            if int(key_cnt.value) == 0 or key_cnt.value < len(key_index):
                continue

            # collect the indices of stations that contain 3-channel waveforms
            key_index_diff = dict(cur_key_index.items() - prev_key_index.items())
            for k, v in key_index_diff.items():
                # print('k: ', k)
                if k in toPredict_scnl:
                    # print('continue')
                    continue

                if type(toPredict_scnl) != list:
                    toPredict_scnl = toPredict_scnl.tolist()

                tmp = k.split('_')

                if tmp[1][-1] == 'Z':             
                    try:
                        tmp_idx = [cur_key_index[f"{tmp[0]}_{tmp[1][:2]}{i}_{tmp[2]}_{tmp[3]}"] for i in channel_tail]    # for channel = XXZ
                        toPredict_idx.append(tmp_idx)
                    
                        toPredict_scnl.append(f"{tmp[0]}_{tmp[1]}_{tmp[2]}_{tmp[3]}")

                    except Exception as e:
                        continue
                if tmp[1][-1] in ['1', '4', '7']:
                    try:                       
                        tmp_idx = [cur_key_index[f"{tmp[0]}_{tmp[1][:-1]}{i}_{tmp[2]}_{tmp[3]}"] for i in range(int(tmp[1][-1]), int(tmp[1][-1])+3)]    # for Ch1-9
                        # print("tmp_idx",tmp_idx,f"{tmp[0]}_{tmp[1]}_{tmp[2]}_{tmp[3]}")
                        toPredict_idx.append(tmp_idx)

                        toPredict_scnl.append(f"{tmp[0]}_{tmp[1]}_{tmp[2]}_{tmp[3]}")
                        # print(toPredict_idx,toPredict_scnl)
                    except Exception as e:
                        continue

            # skip if there is no station need to predict
            if len(toPredict_idx) == 0:
                continue

            # take only 3000-sample waveform
            start = int(env_config['STORE_LENGTH'])//2-int(env_config['PREDICT_LENGTH'])
            end = int(env_config['STORE_LENGTH'])//2

            toPredict_wave = cur_waveform_buffer[torch.tensor(toPredict_idx, dtype=torch.long)][:, :, start:end].to(device)
            # try:
            #     plot_idx = toPredict_scnl.index("H020_HLZ_TW_--")
            # except:
            #     pass
            toPredict_scnl = np.array(toPredict_scnl)
            
            # get the factor and coordinates of stations
            if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP':
                station_factor_coords, station_list, flag = get_Palert_CWB_coord(toPredict_scnl, stationInfo)

                # count to gal
                factor = torch.tensor([f[-1] for f in station_factor_coords])
                
                toPredict_wave = toPredict_wave/factor[:, None, None].to(device)
            else:
                station_factor_coords, station_list, flag = get_coord_factor(toPredict_scnl, stationInfo)

                # multiply with factor to convert count to 物理量
                factor = np.array([f[-1] for f in station_factor_coords])
                toPredict_wave = toPredict_wave*factor[:, :, None].to(device)

            # preprocess
            # 1) convert traces to acceleration
            # 2) 1-45Hz bandpass filter
            # 3) Z-score normalization
            # 4) calculate features: Characteristic, STA, LTA
           
            # original wave: used for saving waveform
            original_wave = toPredict_wave.clone()
            unnormed_wave = original_wave.clone()
            
            toPredict_wave, _ = z_score(toPredict_wave, env_config['PAD_OPTION'], int(env_config['PAD_PRESIGNAL_LENGTH']))
            if int(env_config['NOISE_KEEP']) != 0:
                _, nonzero_flag = z_score(original_wave.clone(), env_config['PAD_OPTION'], int(env_config['PAD_PRESIGNAL_LENGTH']))
            start = time.time()
            toPredict_wave = filter(toPredict_wave, sos)
            print("filter: ", time.time()-start)
            if env_config["CHECKPOINT_TYPE"] == 'GRADUATE':
                stft = STFT(toPredict_wave).to(device)
                toPredict_wave = calc_feats(toPredict_wave)

            # reduce the result in order to speed up the process
            # only reduce the stations when chunk == -1 
            # print(toPredict_wave[0].cpu().numpy().T)
            # print(toPredict_wave[100].cpu().numpy().T)
            # plt.plot(toPredict_wave[plot_idx].cpu().numpy().T)
            # plt.savefig(f"./tmp/{time.time()}.png")
            # plt.clf()

            # predict
            # print("toPredict_wave",toPredict_wave.shape)           
            with torch.no_grad():
                # for conformer
                if env_config["CHECKPOINT_TYPE"] == 'GRADUATE':
                    out = model(toPredict_wave, stft=stft)[1].detach().squeeze().cpu()   
                # for eqt
                elif env_config["CHECKPOINT_TYPE"] == 'eqt':
                    out = model(toPredict_wave)[1].detach().squeeze().cpu()

            # select the p-arrival time 
            original_res, pred_trigger = evaluation(out, float(env_config["THRESHOLD_PROB"]), int(env_config["THRESHOLD_TRIGGER"]), env_config["THRESHOLD_TYPE"])
            # print(pred_trigger)
            original_res = np.logical_and(original_res, flag).tolist()

            # 寫 original res 的 log 檔
            if np.any(original_res):   
                # calculate Pa, Pv, Pd
                Pa, Pv, Pd, duration = picking_append_info(pavd_sta, toPredict_scnl, original_res, pred_trigger, int(env_config['PREDICT_LENGTH']))

                # calculate p_weight
                P_weight = picking_p_weight_info(out, original_res, env_config['PWEIGHT_TYPE'], (float(env_config['PWEIGHT0']), float(env_config['PWEIGHT1']),float(env_config['PWEIGHT2'])),
                                                    toPredict_wave[:, 0].clone(), pred_trigger)

                # send pick_msg to PICK_RING
                original_pick_msg = gen_pickmsg(station_factor_coords, original_res, pred_trigger, toPredict_scnl, cur_waveform_starttime, (Pa, Pv, Pd), duration, P_weight, int(env_config['STORE_LENGTH']), int(env_config['PREDICT_LENGTH']))

                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                original_picking_logfile = f"./log/original_picking/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

                print('Original pick: ', original_res.count(True))
                # writing original picking log file
                with open(original_picking_logfile,"a") as pif:
                    cur_time = datetime.utcfromtimestamp(time.time())
                    pif.write('='*25)
                    pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                    for msg in original_pick_msg:
                        # print(msg)
                        tmp = msg.split(' ')
                        pif.write(" ".join(tmp[:6]))

                        pick_time = datetime.utcfromtimestamp(float(tmp[-4]))
                        # print('pick_time: ', pick_time)
                        pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                        pif.write(f"{msg}\n")

                        # write pick_msg to PICK_RING
                        # if int(env_config['ORIGINAL_PICK_WRITE_PICKRING']) == 1:
                        #     MyModule.put_bytes(1, int(env_config["PICK_MSG_TYPE"]), bytes(msg, 'utf-8'))

                    pif.close()
                
                # filter the picked station that picked within picktime_gap seconds before
                original_res, pick_record = check_duplicate_pick(original_res, toPredict_scnl, pick_record, pred_trigger, cur_waveform_starttime, int(env_config["PICK_GAP"]), int(env_config['STORE_LENGTH']), int(env_config['PREDICT_LENGTH']))

                # print(pick_record)
                # 檢查 picking time 是否在 2500-th sample 之後
                original_res, pred_trigger, res = EEW_pick(original_res, pred_trigger, int(env_config['VALID_PICKTIME']))

                # print(pred_trigger)
                # 區域型 picking
                if int(env_config['AVOID_FP']) == 1:
                    if env_config['TABLE_OPTION'] == 'nearest':
                        res = neighbor_picking(neighbor_table, station_list, res, original_res, int(env_config['THRESHOLD_NEIGHBOR']))   # 用鄰近測站來 pick
                    elif env_config['TABLE_OPTION'] == 'km':
                        res = post_picking(station_factor_coords, res, float(env_config["THRESHOLD_KM"]))                         # 用方圓幾公里來 pick

                # calculate Pa, Pv, Pd
                Pa, Pv, Pd, duration = picking_append_info(pavd_sta, toPredict_scnl, res, pred_trigger, int(env_config['PREDICT_LENGTH']), clear=True)

                # calculate p_weight
                P_weight = picking_p_weight_info(out, res, env_config['PWEIGHT_TYPE'], (float(env_config['PWEIGHT0']), float(env_config['PWEIGHT1']),float(env_config['PWEIGHT2'])),
                                                    toPredict_wave[:, 0].clone(), pred_trigger)

                # send pick_msg to PICK_RING
                pick_msg = gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, cur_waveform_starttime, (Pa, Pv, Pd), duration, P_weight, int(env_config['STORE_LENGTH']), int(env_config['PREDICT_LENGTH']))
                # print("pick_msg",pick_msg)
                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                picking_logfile = f"./log/picking/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}_picking_chunk{env_config['CHUNK']}.log"
                original_picking_logfile = f"./log/original_picking/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

                # writing picking log file
                picked_coord = []
                notify_msg = []
                with open(picking_logfile,"a") as pif:
                    cur_time = datetime.utcfromtimestamp(time.time())
                    pif.write('='*25)
                    pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                    for msg in pick_msg:
                        #print(msg)
                        tmp = msg.split(' ')

                        # filtered by P_weight
                        p_weight = int(tmp[-3])

                        if p_weight <= int(env_config['REPORT_P_WEIGHT']):
                            pif.write(" ".join(tmp[:6]))

                            pick_time = datetime.utcfromtimestamp(float(tmp[-4]))
                            # print('pick_time: ', pick_time)
                            pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                            pif.write(f"{msg}\n")
                            # report(msg , 'EQTransformer')

                            # write pick_msg to PICK_RING
                            # if int(env_config['PICK_WRITE_PICKRING']) == 1:
                            #     MyModule.put_bytes(1, int(env_config["PICK_MSG_TYPE"]), bytes(msg, 'utf-8'))

                            picked_coord.append((float(tmp[4]), float(tmp[5])))
                            notify_msg.append(msg)
                    pif.close() 

                # plotting the station on the map and send info to Line notify
                cur_time = datetime.utcfromtimestamp(time.time())
                print(f"{len(picked_coord)} stations are picked! <- {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                # report = f"{len(picked_coord)} stations are picked! <- {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
                # report(report)
                if len(picked_coord) >= int(env_config["REPORT_NUM_OF_TRIGGER"]):
                    # write line notify info into log file
                    picking_logfile = f"./log/notify/{env_config['CHECKPOINT_TYPE']}_{cur_time.year}-{cur_time.month}-{cur_time.day}_notify_chunk{env_config['CHUNK']}.log"
                    with open(picking_logfile,"a") as pif:
                        cur_time = datetime.utcfromtimestamp(time.time())
                        pif.write('='*25)
                        pif.write(f"Notify time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                        pif.write('='*25)
                        pif.write('\n')
                        for picked_idx, msg in enumerate(notify_msg):
                            toNotify_pickedCoord[picked_idx] = picked_coord[picked_idx]

                            # print(msg)
                            tmp = msg.split(' ')
                            pif.write(" ".join(tmp[:6]))

                            pick_time = datetime.utcfromtimestamp(float(tmp[-5]))
                            pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                            pif.write(f"{msg}\n")
                            # report(msg , 'EQTransformer')
                        pif.close()
                        
                    # send signal for Notifier to send Line notify
                    if int(env_config['LINE_NOTIFY']) == 1:
                        n_notify.value *= 0
                        n_notify.value += len(picked_coord)
                        notify_TF.value += 1

                        # plot the waveform and the prediction
                        isPlot = True

                # 將 picked stations 的 picking time 加進 pick_stat 做統計
                pick_stat += new_pred_trigger

                # 將 picked stations 的 picking time 加進 pick_stat 做統計
                if len(new_pred_trigger) >= int(env_config["REPORT_NUM_OF_TRIGGER"]):
                    pick_stat_notify.value += len(new_pred_trigger)

                # Let's send line notify for one of the picked stations
                if isPlot:
                    idx = np.arange(len(res))
                    tri_idx = idx[res]

                    plot_idx = random.sample(range(len(tri_idx)), k=1)[0]
                    
                    if int(P_weight[plot_idx]) > int(env_config['REPORT_P_WEIGHT']):
                        pass
                    else:
                        plot_info.value = pick_msg[plot_idx]
                        waveform_plot_wf[0] = toPredict_wave[tri_idx[plot_idx], :3].cpu()
                        waveform_plot_out[0] = out[tri_idx[plot_idx]].detach()
                        waveform_plot_picktime.value = pred_trigger[tri_idx[plot_idx]]
                        waveform_plot_TF.value += 1

            else:
                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                original_picking_logfile = f"./log/original_picking/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

                # writing original picking log file
                with open(original_picking_logfile,"a") as pif:
                    cur_time = datetime.utcfromtimestamp(time.time())
                    pif.write('='*25)
                    pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                    pif.close()

                # plt.plot(toPredict_wave[0].cpu().numpy().T)
                # plt.savefig(f"./tmp/{time.time()}.png")
                # plt.clf()
                print(f"(else)0 stations are picked! <- {cur}")   

            prev_key_index = cur_key_index

            # pending until 1 second
            while True:
                cur = time.time()
                if cur - system_record_time >= 0.75:
                    break

        except Exception as e:
            # log the pending 
            # print(e)
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Picker): {e}\n")
                pif.write(f"Trace back (Picker): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()

            continue
        # return report

# notifing 
def Notifier(notify_TF, toNotify_pickedCoord, line_tokens, n_notify, chunk, CHECKPOINT_TYPE):
    print('Starting Notifier...')

    token_number, line_token_number = 0, 0
    while True:
        # pending until notify_TF = True, then send notify
        if notify_TF.value == 0.0 or n_notify.value == 0:
            continue
        
        try:
            picked_coord = []
            for i in range(int(n_notify.value)):
                picked_coord.append(toNotify_pickedCoord[i])

            cur_time = datetime.utcfromtimestamp(time.time())
            trigger_plot_filename = f"{cur_time.year}-{cur_time.month}-{cur_time.day}_{cur_time.hour}:{cur_time.minute}:{cur_time.second}"
            
            start = time.time()
            line_token_number = plot_taiwan(trigger_plot_filename, picked_coord, line_tokens, line_token_number)
            #line_token_number = plot_taiwan_pygmt(trigger_plot_filename, picked_coord, line_tokens, line_token_number)
            notify_TF.value *= 0

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{CHECKPOINT_TYPE}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Notifier): {e}\n")
                pif.write(f"Trace back (Shower): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()

            notify_TF.value *= 0
            continue

# plotting
def Shower(waveform_plot_TF, plot_info, waveform_plot_wf, waveform_plot_out, waveform_plot_picktime, waveform_tokens, CHECKPOINT_TYPE):
    print('Starting Shower...')

    token_number = 0
    while True:
        # don't save the trace, keep pending ...
        if waveform_plot_TF.value == 0.0:
            continue
            
        try:            
            cur = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            
            tmp = plot_info.value.split(' ')

            # save waveform into png files
            scnl = "_".join(tmp[:4])
            savename = f"{cur}_{scnl}"
            png_filename = f"./plot/{savename}.png"

            # png title
            first_title = "_".join(tmp[:6])
            second_title = "_".join(tmp[6:10])
            p_arrival = datetime.utcfromtimestamp(float(tmp[10])).strftime('%Y-%m-%d %H:%M:%S.%f')
            other_title = "_".join(tmp[11:])
            title = f"{first_title}\n{second_title}\n{p_arrival}\n{other_title}"

            plt.figure(figsize=(12, 18))
            plt.rcParams.update({'font.size': 18})
        
            plt.subplot(411)
            plt.plot(waveform_plot_wf[0, 0])
            plt.axvline(x=waveform_plot_picktime.value, color='r')
            plt.title(title)
            
            plt.subplot(412)
            plt.plot(waveform_plot_wf[0, 1])
            plt.axvline(x=waveform_plot_picktime.value, color='r')

            plt.subplot(413)
            plt.plot(waveform_plot_wf[0, 2])
            plt.axvline(x=waveform_plot_picktime.value, color='r')

            plt.subplot(414)
            plt.ylim([-0.05, 1.05])
            plt.axvline(x=waveform_plot_picktime.value, color='r')
            plt.plot(waveform_plot_out[0])
            
            plt.savefig(png_filename)
            plt.clf()
            plt.close('all')

            token_number = random.sample(range(len(waveform_tokens)), k=1)[0]
            token_number = plot_notify(png_filename, waveform_tokens, token_number)
                    
            os.remove(png_filename)
            waveform_plot_TF.value *= 0

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{CHECKPOINT_TYPE}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Shower): {e}\n")
                pif.write(f"Trace back (Shower): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            waveform_plot_TF.value *= 0
            continue

# Upload to google drive
def Uploader(logfilename_pick, logfilename_notify, logfilename_original_pick, logfilename_cwb_pick, logfilename_stat, trc_dir, upload_TF, 
            avg_pickingtime, median_pickingtime, n_pick, cwb_avg_pickingtime, cwb_median_pickingtime, cwb_n_pick, calc_cwbstat, CHECKPOINT_TYPE, pick_stat_notify):
    print('Starting Uploader...')

    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")
    drive = GoogleDrive(gauth)

    while True:
        if upload_TF.value == 0.0:
            continue

        try:
            # 通知 CWBPicker 要開始統計數據
            calc_cwbstat.value += 1.0

            # =========================== picking log ============================= #
            # upload picking log file
            if not os.path.exists(logfilename_pick.value):
                Path(logfilename_pick.value).touch()
            
            file1 = drive.CreateFile({"title":logfilename_pick.value,"parents": [{"kind": "drive#fileLink", "id": "1Y2o_Pp6np8xnxl0QysU4-zVEm4mWOI6_"}]})
            file1.SetContentFile(logfilename_pick.value)
            file1.Upload() #檔案上傳
            print("picking log file -> uploading succeeded!")
            # =========================== picking log ============================= #

            # =========================== original picking ============================ #
            # upload original picking log file
            if not os.path.exists(logfilename_original_pick.value):
                Path(logfilename_original_pick.value).touch()

            file1 = drive.CreateFile({"title":logfilename_original_pick.value,"parents": [{"kind": "drive#fileLink", "id": "1QmeQbsyjajpKHQXcuNxjNm426J--GZ15"}]})
            file1.SetContentFile(logfilename_original_pick.value)
            file1.Upload() #檔案上傳
            print("original picking log file -> uploading succeeded!")
            # =========================== original picking ============================ #

            # ============================== notify log ================================ #
            # upload notify log file
            if not os.path.exists(logfilename_notify.value):
                Path(logfilename_notify.value).touch()

            file1 = drive.CreateFile({"title":logfilename_notify.value,"parents": [{"kind": "drive#fileLink", "id": "1aqLRskDjn7Vi7WB-uzakLiBooKSe67BD"}]})
            file1.SetContentFile(logfilename_notify.value)
            file1.Upload() #檔案上傳
            print("notify log file -> uploading succeeded!")
            # ============================== notify log ================================ #

            # =========================== cwbpicker log ============================ #
            # upload cwbpicker log file
            if not os.path.exists(logfilename_cwb_pick.value):
                Path(logfilename_cwb_pick.value).touch()

            file1 = drive.CreateFile({"title":logfilename_cwb_pick.value,"parents": [{"kind": "drive#fileLink", "id": "1w35MfnWE3em1I0Whrd-cFn64LRKxPMBc"}]})
            file1.SetContentFile(logfilename_cwb_pick.value)
            file1.Upload() #檔案上傳
            print("CWB picker log file -> uploading succeeded!")
            # =========================== cwbpicker log ============================ #

            # upload picked waveform png file 
            cur = datetime.fromtimestamp(time.time())
            cur = cur - timedelta(days=1)
            year, month, day = int(cur.year), int(cur.month), int(cur.day)
            noise_month = str(month) if int(month) >= 10 else f"0{month}"
            if day <= 7:
                dirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week1_picked_check"
                cwbdirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week1_CWBpicked_check"
                noise_png_path = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week1_noise_check_figures"
                noise_pt_pathf = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week1_noise_check_tensors"
            elif day <= 14 and day > 7:
                dirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week2_picked_check"
                cwbdirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week2_CWBpicked_check"
                noise_png_path = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week2_noise_check_figures"
                noise_pt_pathf = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week2_noise_check_tensors"
            elif day <= 21 and day > 14:
                dirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week3_picked_check"
                cwbdirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week3_CWBpicked_check"
                noise_png_path = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week3_noise_check_figures"
                noise_pt_pathf = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week3_noise_check_tensors"
            elif day > 21:
                dirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week4_picked_check"
                cwbdirname = f"{CHECKPOINT_TYPE}_{year}_{month}_week4_CWBpicked_check"
                noise_png_path = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week4_noise_check_figures"
                noise_pt_pathf = f"{CHECKPOINT_TYPE}_{year}_{noise_month}_week4_noise_check_tensors"

            # =========================== exception log ============================ #
            # 上傳 exception log
            exception_log = f"./log/exception/{CHECKPOINT_TYPE}_{year}-{month}-{day}.log"
            if not os.path.exists(exception_log):
                Path(exception_log).touch()

            file1 = drive.CreateFile({"title":exception_log,"parents": [{"kind": "drive#fileLink", "id": "1WwvVw3FGZtnk1hNfK1Mss8iOwuWsUJ2w"}]})
            file1.SetContentFile(exception_log)
            file1.Upload() #檔案上傳
            print("Exception log file -> uploading succeeded!")
            # =========================== exception picking ============================ #

            # =========================== picked waveform png ============================ #
            try:
                if os.path.exists(f"./plot/{dirname}"):
                    # 先將整個資料夾 zip 壓縮
                    archived = shutil.make_archive(f"./plot/{dirname}", 'zip', f"./plot/{dirname}")

                    zip_filename = f"./plot/{dirname}.zip"
                    file1 = drive.CreateFile({"title":zip_filename,"parents": [{"kind": "drive#fileLink", "id": "1ibIVlipURJGSIQv_FlTQcFdodCeDtehE"}]})
                    file1.SetContentFile(zip_filename)
                    file1.Upload() #檔案上傳
                    print(f"PickedWaveform zip file {dirname} -> uploading succeeded!")
                    
                    os.remove(zip_filename)
            except:
                pass

            try:
                if os.path.exists(f"./plot/{cwbdirname}"):
                    # 先將整個資料夾 zip 壓縮
                    archived = shutil.make_archive(f"./plot/{cwbdirname}", 'zip', f"./plot/{cwbdirname}")

                    zip_filename = f"./plot/{cwbdirname}.zip"
                    file1 = drive.CreateFile({"title":zip_filename,"parents": [{"kind": "drive#fileLink", "id": "121tJrHxmQpl7BUWiH_oqPpMcTaPkZ_xS"}]})
                    file1.SetContentFile(zip_filename)
                    file1.Upload() #檔案上傳
                    print(f"CWBPickedWaveform zip file {cwbdirname} -> uploading succeeded!")

                    os.remove(zip_filename)
            except:
                pass
            # =========================== picked waveform png ============================ #

            # =========================== noise waveform png & pt ============================ #
            try:
                # 將 noise sample 與 png 上傳
                if os.path.exists(f"./plot/{noise_png_path}"):
                    # 先將整個資料夾 zip 壓縮
                    archived = shutil.make_archive(f"./plot/{noise_png_path}", 'zip', f"./plot/{noise_png_path}")

                    zip_filename = f"./plot/{noise_png_path}.zip"
                    file1 = drive.CreateFile({"title":zip_filename,"parents": [{"kind": "drive#fileLink", "id": "10fZP97Lnv259ki_mLX-Ibu9d85mnPk6J"}]})
                    file1.SetContentFile(zip_filename)
                    file1.Upload() #檔案上傳
                    print(f"Noise figures zip file {noise_png_path} -> uploading succeeded!")

                    os.remove(zip_filename)

                if os.path.exists(f"./plot/{noise_pt_pathf}"):
                    # 先將整個資料夾 zip 壓縮
                    archived = shutil.make_archive(f"./plot/{noise_pt_pathf}", 'zip', f"./plot/{noise_pt_pathf}")

                    zip_filename = f"./plot/{noise_pt_pathf}.zip"
                    file1 = drive.CreateFile({"title":zip_filename,"parents": [{"kind": "drive#fileLink", "id": "10fZP97Lnv259ki_mLX-Ibu9d85mnPk6J"}]})
                    file1.SetContentFile(zip_filename)
                    file1.Upload() #檔案上傳
                    print(f"Noise pytorch tensors zip file {noise_pt_pathf} -> uploading succeeded!")

                    os.remove(zip_filename)
            except:
                pass
            # =========================== noise waveform png & pt ============================ #

            # =========================== statistical ============================ #
            try:
                # 將 AI picker 昨日的 picking 統計數據寫成 log 檔
                with open(logfilename_stat.value, 'a') as f:
                    f.write('='*50)
                    f.write('\n')
                    f.write(f"{cur.year}-{cur.month}-{cur.day} AI picker statistical\n")
                    f.write(f"Number of picked stations: {n_pick.value}\n")
                    f.write(f"Number of picked stations in Line Notify: {pick_stat_notify.value}\n")
                    f.write(f"Average picking time that model need (delay): {avg_pickingtime.value} s\n")
                    f.write(f"Median picking time that model need (delay): {median_pickingtime.value} s\n")
                    f.write('='*50)
                    f.write('\n')
                
                # 將 CWB picker 昨日的 picking 統計數據寫成 log 檔
                while True:
                    if calc_cwbstat.value == 0.0:
                        break

                with open(logfilename_stat.value, 'a') as f:
                    f.write('='*50)
                    f.write('\n')
                    f.write(f"{cur.year}-{cur.month}-{cur.day} CWB picker statistical\n")
                    f.write(f"Number of picked stations: {cwb_n_pick.value}\n")
                    f.write(f"Average picking time that model need (delay): {cwb_avg_pickingtime.value} s\n")
                    f.write(f"Median picking time that model need (delay): {cwb_median_pickingtime.value} s\n")
                    f.write('='*50)
                    f.write('\n')

                if not os.path.exists(logfilename_stat.value):
                    Path(logfilename_stat.value).touch()

                pick_stat_notify.value *= 0

                file1 = drive.CreateFile({"title":logfilename_stat.value,"parents": [{"kind": "drive#fileLink", "id": "1SO8DUFMshG2E2P33KCx7146cBR0V0jXG"}]})
                file1.SetContentFile(logfilename_stat.value)
                file1.Upload() #檔案上傳
                print("statistical file -> uploading succeeded!")
            except:
                pass
            # =========================== statistical ============================ #

            upload_TF.value *= 0.0
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{CHECKPOINT_TYPE}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Uploader): {e}\n")
                pif.write(f"Trace back (Uploader): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            upload_TF.value *= 0.0

# remove all event in ./plot and ./trace
def Remover(remove_daily, expired_day, CHECKPOINT_TYPE):
    print('Starting Remover...')

    while True:
        # 檢查是不是已經試過了一天
        if remove_daily.value == 0:
            continue

        try:
            # 過期，刪除所有在 ./plot & ./trace 的事件資料夾
            folders = os.listdir('./trace')

            cur = datetime.fromtimestamp(time.time())
            cur_year, cur_month, cur_day = cur.year, cur.month, cur.day
            cur_day = int(cur_day)
            if cur_day <= 7:
                cur_day = 1
            elif cur_day <= 14 and cur_day > 7:
                cur_day = 7
            elif cur_day <= 21 and cur_day > 14:
                cur_day = 14
            elif cur_day > 21:
                cur_day = 21
            cur = datetime(year=int(cur_year), month=int(cur_month), day=cur_day)

            for f in folders:
                ymw = f.split('_')
                f_day = ymw[-1]
                if f_day == 'week4':
                    f_day = 21
                elif f_day == 'week3':
                    f_day = 14
                elif f_day == 'week2':
                    f_day = 7
                else:
                    f_day = 1
                folder_date = datetime(year=int(ymw[1]), month=int(ymw[2]), day=f_day)
                
                if (cur-folder_date).days >= int(expired_day):
                    shutil.rmtree(f"./trace/{f}")
                    shutil.rmtree(f"./plot/{f}") 

            remove_daily.value *= 0
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{CHECKPOINT_TYPE}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Remover): {e}\n")
                pif.write(f"Trace back (Remover): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            remove_daily.value *= 0

# Calculating the Pa, Pv, and Pd per sample, only Z-component will be calculated
def Pavd_calculator(pavd_calc, waveform_comein, waveform_scnl, waveform_comein_length, pavd_sta, stationInfo, env_config):
    print('Starting Pavd_calculator...')
    MyManager.register('EEW_sta', EEW_sta)
    
    manager = Manager_Pavd()
    while True:
        try:
            if pavd_calc.value == 0:
                continue

            cur_waveform_scnl = waveform_scnl.value
            nsamp = int(waveform_comein_length.value)
            wf = waveform_comein[0][:nsamp]
        
            # new station come in, creating the Pavd object
            if cur_waveform_scnl not in pavd_sta:
                tmp = cur_waveform_scnl.split('_')

                # get gain factor
                if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP':
                    station_factor_coords, _, _ = get_Palert_CWB_coord([cur_waveform_scnl], stationInfo)

                    station_factor_coords = float(station_factor_coords[0][-1])

                else:
                    station_factor_coords, _, _ = get_coord_factor([cur_waveform_scnl], stationInfo)
                    station_factor_coords = 1/station_factor_coords[0][-1][0]

                # pavd_sta[cur_waveform_scnl] = manager.EEW_sta(tmp[0], tmp[1], tmp[2], tmp[3], station_factor_coords)
                pavd_sta[cur_waveform_scnl] = {}
                pavd_sta[cur_waveform_scnl] = create_station(pavd_sta[cur_waveform_scnl], station_factor_coords)
                
            # calculate the station's A, V, and D
            # Step1: Demean
            wf = wf - torch.mean(wf)   

            # Step2: Calculate through object's API
            # pavd_sta[cur_waveform_scnl].receive_new_waveform(wf.numpy().tolist(), nsamp)
            pavd_sta[cur_waveform_scnl] = update_avd(pavd_sta[cur_waveform_scnl], wf.numpy().tolist(), nsamp)
            
            pavd_calc.value *= 0
            waveform_comein_length.value *= 0

        except Exception as e:
            # log the pending 
            # print(e)
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{env_config['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Pavd_calculator): {e}\n")
                pif.write(f"Trace back (Pavd_calculator): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()

            pavd_calc.value *= 0
            waveform_comein_length.value *= 0
            continue

def PavdModule_sender(CHECKPOINT_TYPE, pavd_calc, waveform_comein, waveform_comein_length, pavd_scnl, waveform_scnl):
    
    # Send information into first Pavd module if idle
    while True:
        try:
            w = pavd_scnl.get()

            if w == {}:
                continue

            for k, v in w.items():
                scnl = k
                waveform = v
                nsamp = 100

            for i in range(7):
                if pavd_calc[i].value == 0:
                    waveform_scnl[i].value = scnl
                    waveform_comein[i][0][:nsamp] = torch.from_numpy(waveform)
                    waveform_comein_length[i].value += nsamp   
                    pavd_calc[i].value += 1    

                    break     
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{CHECKPOINT_TYPE}_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (PavdModule_sender): {e}\n")
                pif.write(f"Trace back (PavdModule_sender): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            pass

def create_dir():
    if not os.path.exists('./log/picking'):
        os.makedirs('./log/picking')
    if not os.path.exists('./log/notify'):
        os.makedirs('./log/notify')
    if not os.path.exists('./log/CWBPicker'):
        os.makedirs('./log/CWBPicker')
    if not os.path.exists('./log/original_picking'):
        os.makedirs('./log/original_picking')
    if not os.path.exists('./log/exception'):
        os.makedirs('./log/exception')
    if not os.path.exists('./log/statistical'):
        os.makedirs('./log/statistical')
    if not os.path.exists('./plot'):
        os.makedirs('./plot')
    if not os.path.exists('./plot/trigger'):
        os.makedirs('./plot/trigger')
    if not os.path.exists('./trace'):
        os.makedirs('./trace')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force = True)
    
    # create the directories
    create_dir()
       
    # start the system
    # including mqtt, picker, ... modules
    mqttserver = Mqtt()
    mqttserver.start()
      