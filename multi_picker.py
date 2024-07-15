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
from tqdm import tqdm

import ctypes as c
import random
import pandas as pd 
import os
import glob
import bisect
import shutil
import uuid
import datetime
import sys
import gc

from ctypes import c_char_p
from dotenv import dotenv_values
from datetime import datetime, timedelta, timezone
from collections import deque
from picking_preprocess import *
from picking_utils import *
from picking_model import *
from Pavd_module import *

import seisbench.models as sbm

# time consuming !
import matplotlib.pyplot as plt
import json
import paho.mqtt.client as mqtt
from obspy import read
import struct

# For RED-PAN
sys.path.append('./redpan')
import tensorflow as tf
from REDPAN_tools.mtan_ARRU import unets

# for shared lass object
class MyManager(BaseManager): pass
def Manager_Pavd():
    m = MyManager()
    m.start()
    return m 

def station_chunk_per_picker(source, n_prediction_station, stationInfo, channel, location):
    # Slice the chunk depends on the stationInfo
    source = self.env_config['SOURCE']
    n_stations = int(self.env_config['N_PREDICTION_STATION'])
    if source == 'Palert':
        station_chunks = ForPalert_station_selection(self.stationInfo, n_stations)
    elif source == 'TSMIP':
        station_chunks = ForTSMIP_station_selection(self.stationInfo)
    # TODO: CWASN

    # Check the chunk number
    chunk = self.env_config['CHUNK']
    end_chunk = None
    if len(chunk) > 2:
        start_chunk = int(chunk[0])
        end_chunk = int(chunk[-1])
    else:
        start_chunk = int(chunk)

        if start_chunk == -1:
            end_chunk = len(station_chunks)-1
        else:
            end_chunk = start_chunk+1
    
    # Merge the chunk, generating the topic name
    scnl = []

    # topic template: source/network/station/location/channel
    # ex. CWASN24Bit/TW/ALS/10/HLZ
    if self.env_config['CHANNEL'] != 'None':
        channel_tail = self.env_config['CHANNEL'].split(',')
    else:
        channel_tail = ['Z', 'N', 'E']

    location_list = []
    if self.env_config['LOCATION'] != 'None':
        location_list = self.env_config['LOCATION'].split(',')

    for ch in range(start_chunk, end_chunk):
        # station_chunks => station_code: [lontitude, latitude, factor, channel, location]
        for sta in station_chunks[ch]:
            for chn in channel_tail:
                # filter the location
                if len(location_list) == 0 or sta[1][4] in location_list:
                    scnl.append(f"{sta[0]}_{sta[1][3]}{chn}_TW_{sta[1][4]}")
    return scnl

class Mqtt_Predictor():
    def __init__(self,):
        
        # initialize the shared variables
        self.init_shared_params()
        
        # station selection
        self.station_chunk()

        # start the mqtt 
        self.activate_mqtt()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code " + str(rc))
    
            for t in self.topic:
                client.subscribe(t)

            cur = datetime.fromtimestamp(time.time())
            with open(f"./log/mqtt/{cur.year}_{cur.month}_{cur.day}.log", 'a') as f:
                f.write(f"MQTT connected at {cur.hour}:{cur.minute}:{cur.second}\n")
                f.write('='*25)
                f.write('\n')
        else:
            print("Failed to connect, ", rc)
            client.disconnect()

    def on_message(self, client, userdata, msg):
        # parse the package: 0.0003 s on average
        msg = msg.payload
        network, station, location, channel, nsamp, samprate, starttime, endtime = struct.unpack(f"<2s5s2s3sIddd", msg[0:40])
        network = network.decode().strip()
        station = station.decode().strip()
        location = location.decode().strip()
        channel = channel.decode().strip()
        scnl = f"{station}_{channel}_{network}_{location}"
        data = np.array(struct.unpack(f"<{nsamp:d}i", msg[40:]), dtype=np.float32)

        # update the key_index
        if scnl not in self.key_index:
            if scnl not in self.key_index:
                self.key_index[scnl] = int(self.key_cnt.value)
                self.key_cnt.value += 1

        # append the data in package into shared waveform buffer
        startIndex = int(starttime*self.samp_rate) - int(self.waveform_buffer_start_time.value)
        
        if startIndex >= 0 and startIndex < self.store_length:
            data = data.copy()
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

            cur = datetime.fromtimestamp(time.time())
            with open(f"./log/mqtt/{cur.year}_{cur.month}_{cur.day}.log", 'a') as f:
                f.write(f"MQTT disconnected at {cur.hour}:{cur.minute}:{cur.second}\n")
                f.write('='*25)
                f.write('\n')

    def activate_mqtt(self):
        # 建立 MQTT Client 物件
        self.client = mqtt.Client()
        # 設定建立連線回呼函數 (callback function)
        self.client.on_connect = self.on_connect

        # callback functions
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect    

        # 連線至 MQTT 伺服器（伺服器位址,連接埠）, timeout=6000s
        self.client.connect(self.env_config['MQTT_SERVER'], int(self.env_config['PORT']), int(self.env_config['TIMEOUT']))
        print("connect!")
        
        # 進入無窮處理迴圈
        # self.client.loop_forever()
        self.client.loop_start()

    def activate_single_picker(self, picker):
        proc = None
        if int(self.picker[1]) == 1:
            # device
            device = torch.device(self.picker[2] if torch.cuda.is_available() else "cpu")  
            
            # taking shared env params as local params
            local_env = {}
            for k, v in self.env_config.items():
                local_env[k] = v

            # updating local params with seperate env file of each picker
            for k, v in dotenv_values(self.picker[0]).items():
                local_env[k] = v

            picker = Process(target=Picker, args=(self.waveform_buffer, self.key_index, self.nowtime, self.waveform_buffer_start_time, local_env, self.key_cnt, self.stationInfo, device,
                                                (self.notify_tokens, self.waveform_tokens), self.pavd_sta))
                
            return proc

    def start(self):
        try:
            time_mover = Process(target=TimeMover, args=(self.waveform_buffer, self.env_config, self.nowtime, self.waveform_buffer_start_time))
            time_mover.start()

            pavd_sender = Process(target=PavdModule_sender, args=(self.env_config['CHECKPOINT_TYPE'], int(self.env_config['SAMP_RATE']), self.pavd_calc, self.waveform_comein, self.waveform_comein_length, self.pavd_scnl, self.waveform_scnl))
            pavd_sender.start()

            picker = []
            for e in [self.eqt_env, self.phasenet_env, self.graduate_env, self.redpan_env, self.stalta_env]:
                cur_picker = self.activate_single_picker(e)

                if cur_picker is not None:
                    cur_picker.start()
                    picker.append(cur_picker)

            pavd_processes = []
            for i in range(int(self.env_config['N_PAVD_PROCESS'])):
                pavd_calculator = Process(target=Pavd_calculator, args=(self.pavd_calc[i], self.waveform_comein[i], self.waveform_scnl[i], self.waveform_comein_length[i], self.pavd_sta, self.stationInfo, self.env_config))
                pavd_calculator.start()
                pavd_processes.append(pavd_calculator)     

            time_mover.join()
            pavd_sender.join()

            for p in picker:
                p.join()
            
            for w in pavd_processes:
                w.join()
            
        except KeyboardInterrupt:
            time_mover.terminate()
            pavd_sender.terminate()
            for p in picker:
                p.terminate()
                p.join()

            for w in pavd_processes:
                w.terminate()
                w.join()
    
            time_mover.join()
            pavd_sender.join()
            picker.join()

    def init_shared_params(self):
        # create multiprocessing manager to maintain the shared variables
        manager = Manager()
        
        self.env_config = manager.dict()
        for k, v in dotenv_values(sys.argv[1]).items():
            self.env_config[k] = v
            
        self.eqt_env = [self.env_config['EQT_ENV'], self.env_config['EQT'], self.env_config['EQT_DEVICE']]
        self.phasenet_env = [self.env_config['phaseNet_ENV'], self.env_config['phaseNet'], self.env_config['phaseNet_DEVICE']]
        self.graduate_env = [self.env_config['GRADUATE_ENV'], self.env_config['GRADUATE'], self.env_config['GRADUATE_DEVICE']]
        self.redpan_env = [self.env_config['REDPAN_ENV'], self.env_config['REDPAN'], self.env_config['REDPAN_DEVICE']]
        self.stalta_env = [self.env_config['STALTA_ENV'], self.env_config['STALTA'], self.env_config['STALTA_DEVICE']]

        self.samp_rate = int(self.env_config['SAMP_RATE'])
        self.store_length = int(self.env_config['STORE_LENGTH'])

        # get the candidate line notify tokens
        self.notify_tokens, self.waveform_tokens = load_tokens(self.env_config['NOTIFY_TOKENS'], self.env_config['WAVEFORM_TOKENS'])
        
        # get the station's info
        if self.env_config['SOURCE'] == 'Palert':
            self.stationInfo = get_PalertStationInfo(self.env_config['STATION_FILEPATH'])
        elif self.env_config['SOURCE'] == 'CWASN':
            self.stationInfo = get_CWBStationInfo(self.env_config['STATION_FILEPATH'])
        elif self.env_config['SOURCE'] == 'TSMIP':
            self.stationInfo = get_TSMIPStationInfo(self.env_config['STATION_FILEPATH'])
        else:
            self.stationInfo = get_StationInfo(self.env_config['STATION_FILEPATH'], (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))

        chunk = self.env_config['CHUNK']
        if len(chunk) > 2:
            n_chunk = int(chunk[-1]) - int(chunk[0]) + 1
            n_stations = n_chunk * int(self.env_config["N_PREDICTION_STATION"])
        else:
            n_stations = int(self.env_config["N_PREDICTION_STATION"])

        # a deque from time-3000 to time for time index
        self.nowtime = Value('d', int(time.time()*100))
        self.waveform_buffer_start_time = Value('d', self.nowtime.value-3000)

        # a counter for accumulating key's count
        self.key_cnt = Value('d', int(0))

        # a dict for checking scnl's index of waveform
        self.key_index = manager.dict()
        self.waveform_buffer = torch.zeros((n_stations*3, int(self.env_config["STORE_LENGTH"]))).share_memory_()

        # parameters for Pavd_calculator
        self.pavd_sta = manager.dict()
        self.pavd_calc = [Value('d', int(0)) for _ in range(7)]
        self.waveform_comein = [torch.empty((1, 500)).share_memory_() for _ in range(7)]
        self.waveform_comein_length = [Value('d', int(0)) for _ in range(7)]
        self.waveform_scnl = [manager.Value(c_char_p, 'hello') for _ in range(7)]
        self.pavd_scnl = manager.Queue()

    def station_chunk(self):
        '''
        Three patterns:
        
        1. CHUNK=-1, taking all of the station to predict
        2. CHUNK=0, select the specific chunk of the stations to predict
        3. CHUNK=0-2, select multiple chunks of the stations to predict
        '''
        # Slice the chunk depends on the stationInfo
        source = self.env_config['SOURCE']
        n_stations = int(self.env_config['N_PREDICTION_STATION'])
        if source == 'Palert':
            station_chunks = ForPalert_station_selection(self.stationInfo, n_stations)
        elif source == 'TSMIP':
            station_chunks = ForTSMIP_station_selection(self.stationInfo)
        # TODO: CWASN

        # Check the chunk number
        chunk = self.env_config['CHUNK']
        end_chunk = None
        if len(chunk) > 2:
            start_chunk = int(chunk[0])
            end_chunk = int(chunk[-1])
        else:
            start_chunk = int(chunk)

            if start_chunk == -1:
                end_chunk = len(station_chunks)-1
            else:
                end_chunk = start_chunk+1
       
        # Merge the chunk, generating the topic name
        topic = []

        # topic template: source/network/station/location/channel
        # ex. CWASN24Bit/TW/ALS/10/HLZ
        topic_prefix = self.env_config['WAVE_TOPIC']
        if start_chunk == -1 and self.env_config['CHANNEL'] == 'None' and self.env_config['LOCATION'] == 'None':
            topic.append(f"{topic_prefix}/TW/#")
        else:
            if self.env_config['CHANNEL'] != 'None':
                channel_tail = self.env_config['CHANNEL'].split(',')
            else:
                channel_tail = ['Z', 'N', 'E']

            location_list = []
            if self.env_config['LOCATION'] != 'None':
                location_list = self.env_config['LOCATION'].split(',')
    
            for ch in range(start_chunk, end_chunk):
                # station_chunks => station_code: [lontitude, latitude, factor, channel, location]
                for sta in station_chunks[ch]:
                    for chn in channel_tail:
                        # filter the location
                        if len(location_list) == 0 or sta[1][4] in location_list:
                            topic.append(f"{topic_prefix}/TW/{sta[0]}/{sta[1][4]}/{sta[1][3]}{chn}")

        self.topic = topic

def TimeMover(waveform_buffer, env_config, nowtime, waveform_buffer_start_time):
    print('Starting TimeMover...')

    local_env = {}
    for k, v in env_config.items():
        local_env[k] = v

    move_second = int(local_env['SAMP_RATE']) * int(local_env['SECOND_MOVE_BUFFER'])
    while True:
        try:
            # move the time window of timeIndex and waveform every 5 seconds
            if int(time.time()*int(local_env['SAMP_RATE'])) - nowtime.value >= move_second:
                waveform_buffer[:, 0:int(local_env["STORE_LENGTH"])-move_second] = waveform_buffer[:, move_second:int(local_env["STORE_LENGTH"])]
                
                # the updated waveform is fill in with 0
                waveform_buffer[:, int(local_env["STORE_LENGTH"])-move_second:int(local_env["STORE_LENGTH"])] = torch.zeros((waveform_buffer.shape[0],move_second))
                waveform_buffer_start_time.value += move_second
                nowtime.value += move_second
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{local_env['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}.log"
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
            tokens, pavd_sta):

    print('Starting Picker...')

    scnl_in_chunk = station_chunk_per_picker(local_env['SOURCE'], local_env['N_PREDICTION_STATION'], stationInfo, local_env['CHANNEL'], local_env['LOCATION'])

    # mqtt server for publishing pick_msg
    client = mqtt.Client()
    client.connect(env_config['MQTT_SERVER'], int(env_config['PORT']), int(env_config['TIMEOUT']))

    # loading pretrained picker
    model_path = local_env["PICKER_CHECKPOINT_PATH"]
    if local_env["CHECKPOINT_TYPE"] == 'GRADUATE':
        in_feat = 12
        model = GRADUATE(conformer_class=8, d_ffn=128, nhead=4, enc_layers=2, dec_layers=1, d_model=12, wavelength=int(local_env['PREDICT_LENGTH'])).to(device)
        torch.cuda.empty_cache()
    elif local_env["CHECKPOINT_TYPE"] == 'eqt':
        in_feat = 3
        model = sbm.EQTransformer(in_samples=int(local_env['PREDICT_LENGTH'])).to(device)
    elif local_env['CHECKPOINT_TYPE'] == 'phaseNet':
        model = sbm.PhaseNet(in_channels=3, classes=3, phases='NPS').to(device)
    elif local_env['CHECKPOINT_TYPE'] == 'REDPAN':
        frame = unets(input_size=(int(local_env['PREDICT_LENGTH']), 3))
        model = frame.build_mtan_R2unet(
            model_path, input_size=(int(local_env['PREDICT_LENGTH']), 3)
        )  

    if local_env['CHECKPOINT_TYPE'] not in ['STALTA', 'REDPAN']:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

    # channel name for specific seismographic network
    if local_env['CHANNEL'] != 'None':
        channel_tail = local_env['CHANNEL'].split(',')
    else:
        channel_tail = ['Z', 'N', 'E']

    # butterworth filter
    N=5
    Wn=[1, 10]
    btype='bandpass'
    analog=False
    _filt_args = (N, Wn, btype, analog)
    sos = scipy.signal.butter(*_filt_args, output="sos", fs=int(local_env['SAMP_RATE']))

    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day
    system_hour = cur.hour
    
    # Neighbor_table: 蒐集每個測站方圓 X km 內所有測站的代碼
    _, neighbor_table = station_selection(sel_chunk=local_env["CHUNK"], station_list=stationInfo, opt=local_env['SOURCE'], build_table=True, n_stations=int(local_env["N_PREDICTION_STATION"]), threshold_km=float(local_env['THRESHOLD_KM']),
                                            nearest_station=int(local_env['NEAREST_STATION']), option=local_env['TABLE_OPTION'])

    # sleeping, 讓波型先充滿 noise，而不是 0
    print('pending...')
    for _ in tqdm(range(int(local_env['SLEEP_SECOND']))):
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

    gpu_devices = tf.config.experimental.list_physical_devices('CPU')

    while True:
        try:
            cur = datetime.fromtimestamp(time.time())
            system_record_time = time.time()

            # 每小時發一個 notify，證明系統還活著
            if f"{system_year}-{system_month}-{system_day}-{system_hour}" != f"{cur.year}-{cur.month}-{cur.day}-{cur.hour}":
                wave_token_number = random.sample(range(len(waveform_tokens)), k=1)[0]
                wave_token_number = alive_notify(waveform_tokens, wave_token_number, local_env['CHECKPOINT_TYPE'], local_env['SOURCE'])
                system_hour = cur.hour

            # 已經是系統時間的隔天，檢查有沒有過舊的 log file，有的話將其刪除
            if f"{system_year}-{system_month}-{system_day}" != f"{cur.year}-{cur.month}-{cur.day}":
                toDelete_picking = cur - timedelta(days=int(local_env['DELETE_PICKINGLOG_DAY']))
                toDelete_notify = cur - timedelta(days=int(local_env['DELETE_NOTIFYLOG_DAY']))

                toDelete_picking_filename = f"./log/picking/{local_env['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_picking_chunk{local_env['CHUNK']}.log"
                toDelete_original_picking_filename = f"./log/original_picking/{local_env['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_original_picking_chunk{local_env['CHUNK']}.log"
                toDelete_notify_filename = f"./log/notify/{local_env['CHECKPOINT_TYPE']}_{toDelete_notify.year}-{toDelete_notify.month}-{toDelete_notify.day}_notify_chunk{local_env['CHUNK']}.log"
                toDelete_exception_filename = f"./log/exception/{local_env['CHECKPOINT_TYPE']}_{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}.log"

                if os.path.exists(toDelete_picking_filename):
                    os.remove(toDelete_picking_filename)
                if os.path.exists(toDelete_original_picking_filename):
                    os.remove(toDelete_original_picking_filename)
                if os.path.exists(toDelete_notify_filename):
                    os.remove(toDelete_notify_filename)
                if os.path.exists(toDelete_exception_filename):
                    os.remove(toDelete_exception_filename)

                system_year, system_month, system_day = cur.year, cur.month, cur.day

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
                if f"{tmp[0]}_{tmp[1]}_{tmp[2]}_{tmp[3]}" not in scnl_in_chunk:
                    continue

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
            start = int(local_env['STORE_LENGTH'])//2-int(local_env['PREDICT_LENGTH'])
            end = int(local_env['STORE_LENGTH'])//2

            toPredict_wave = cur_waveform_buffer[torch.tensor(toPredict_idx, dtype=torch.long)][:, :, start:end].to(device)
            toPredict_scnl = np.array(toPredict_scnl)

            # get the factor and coordinates of stations
            if local_env['SOURCE'] == 'Palert' or local_env['SOURCE'] == 'CWASN' or local_env['SOURCE'] == 'TSMIP':
                station_factor_coords, station_list, flag = get_Palert_CWB_coord(toPredict_scnl, stationInfo)

                # count to gal
                factor = torch.tensor([f[-1] for f in station_factor_coords])
                
                toPredict_wave = toPredict_wave/factor[:, None, None].to(device)
            else:
                continue
    
            # preprocess
            # 1) convert traces to acceleration
            # 2) 1-45Hz bandpass filter
            # 3) Z-score normalization
            # 4) calculate features: Characteristic, STA, LTA      
            unnorm_wave = toPredict_wave.clone()      
            if int(local_env['ZSCORE']) == 1:
                toPredict_wave = z_score(toPredict_wave)

            # For STA/LTA, characteristic function
            if local_env['CHECKPOINT_TYPE'] == 'STALTA':
                toPredict_wave = characteristic(toPredict_wave)
            if int(local_env['FILTER']) == 1:
                toPredict_wave = filter(toPredict_wave, sos)

            if local_env["CHECKPOINT_TYPE"] == 'GRADUATE':
                stft = STFT(toPredict_wave.cpu().numpy()).to(device)
                toPredict_wave = calc_feats(toPredict_wave)

            # predict
            with torch.no_grad():
                # for conformer
                if local_env["CHECKPOINT_TYPE"] == 'GRADUATE':
                    out = model(toPredict_wave, stft=stft)[1].detach().squeeze().cpu()   
                # for eqt
                elif local_env["CHECKPOINT_TYPE"] == 'eqt':
                    out = model(toPredict_wave)[1].detach().squeeze().cpu()
                # for phaseNet
                elif local_env['CHECKPOINT_TYPE'] == 'phaseNet':
                    out = model(toPredict_wave)[:, 0].detach().squeeze().cpu()
                # for STA/LTA
                elif local_env['CHECKPOINT_TYPE'] == 'STALTA':
                    original_res, pred_trigger, out = stalta(toPredict_wave.cpu().numpy(), int(local_env['SHORT_WINDOW']), int(local_env['LONG_WINDOW']), float(local_env['THRESHOLD_LAMBDA']))
                # for REDPAN
                elif local_env['CHECKPOINT_TYPE'] == 'REDPAN':
                    toPredict_wave = tf.convert_to_tensor(toPredict_wave.permute(0,2,1).cpu().numpy(), dtype=tf.float32)
                    Picks, _ = model(toPredict_wave)

                    # Picks, _ = model.predict(toPredict_wave.permute(0,2,1).cpu().numpy())
                    original_res, pred_trigger, out = REDPAN_evaluation(Picks.numpy(), float(local_env["THRESHOLD_PROB"]))
     
            # select the p-arrival time         
            if local_env['CHECKPOINT_TYPE'] not in ['STALTA', 'REDPAN']:
                original_res, pred_trigger = evaluation(out, float(local_env["THRESHOLD_PROB"]), int(local_env["THRESHOLD_TRIGGER"]), local_env["THRESHOLD_TYPE"])

            # 寫 original res 的 log 檔
            if np.any(original_res):   
                # calculate Pa, Pv, Pd
                Pa, Pv, Pd, duration = picking_append_info(pavd_sta, toPredict_scnl, original_res, pred_trigger, int(local_env['PREDICT_LENGTH']))

                # calculate p_weight
                P_weight = picking_p_weight_info(out, original_res, local_env['PWEIGHT_TYPE'], (float(local_env['PWEIGHT0']), float(local_env['PWEIGHT1']),float(local_env['PWEIGHT2'])),
                                                    local_env['CHECKPOINT_TYPE'], unnorm_wave[:, 0].clone(), pred_trigger)

                # send pick_msg to PICK_RING
                original_pick_msg = gen_pickmsg(station_factor_coords, original_res, pred_trigger, toPredict_scnl, cur_waveform_starttime, (Pa, Pv, Pd), duration, P_weight, int(local_env['STORE_LENGTH']), int(local_env['PREDICT_LENGTH']))

                # filter the picked station that picked within picktime_gap seconds before
                original_res, pick_record = check_duplicate_pick(original_res, toPredict_scnl, pick_record, pred_trigger, cur_waveform_starttime, int(local_env["PICK_GAP"]), int(local_env['STORE_LENGTH']), int(local_env['PREDICT_LENGTH']))

                # 檢查 picking time 是否在 2500-th sample 之後
                original_res, pred_trigger, res = EEW_pick(original_res, pred_trigger, int(local_env['VALID_PICKTIME']))

                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                original_picking_logfile = f"./log/original_picking/{local_env['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{local_env['CHUNK']}.log"

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

                        # publish to mqtt (pick_msg)
                        message = {
                            "station" : tmp[0],
                            "channel" : tmp[1],
                            "network" : tmp[2],
                            "location" : tmp[3],
                            "longitude" : tmp[4],
                            "latitude" : tmp[5],
                            "pa" : tmp[6],
                            "pv" : tmp[7],
                            "pd" : tmp[8],
                            "tc" : tmp[9],
                            "ptime" : tmp[10],
                            "weight" : tmp[11],
                            "instrument" : tmp[12],
                            "upd_sec" : tmp[13],
                        }
                        client.publish(f"{local_env['PICK_MSG_TOPIC']}/{local_env['CHECKPOINT_TYPE']}", json.dumps(message))

                    pif.close()
                
                # 區域型 picking
                if int(local_env['AVOID_FP']) == 1:
                    if local_env['TABLE_OPTION'] == 'nearest':
                        res = neighbor_picking(neighbor_table, station_list, res, original_res, int(local_env['THRESHOLD_NEIGHBOR']))   # 用鄰近測站來 pick
                    elif local_env['TABLE_OPTION'] == 'km':
                        res = post_picking(station_factor_coords, res, float(local_env["THRESHOLD_KM"]))                         # 用方圓幾公里來 pick

                if np.any(res):
                    # calculate Pa, Pv, Pd
                    Pa, Pv, Pd, duration = picking_append_info(pavd_sta, toPredict_scnl, res, pred_trigger, int(local_env['PREDICT_LENGTH']), clear=True)

                    # calculate p_weight
                    P_weight = picking_p_weight_info(out, res, local_env['PWEIGHT_TYPE'], (float(local_env['PWEIGHT0']), float(local_env['PWEIGHT1']),float(local_env['PWEIGHT2'])),
                                                        local_env['CHECKPOINT_TYPE'], toPredict_wave[:, 0].clone(), pred_trigger)

                    # send pick_msg to PICK_RING
                    pick_msg = gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, cur_waveform_starttime, (Pa, Pv, Pd), duration, P_weight, int(local_env['STORE_LENGTH']), int(local_env['PREDICT_LENGTH']))

                    # get the filenames
                    cur = datetime.fromtimestamp(time.time())
                    picking_logfile = f"./log/picking/{local_env['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}_picking_chunk{local_env['CHUNK']}.log"

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

                            if p_weight <= int(local_env['REPORT_P_WEIGHT']):
                                pif.write(" ".join(tmp[:6]))

                                pick_time = datetime.utcfromtimestamp(float(tmp[-4]))
                                # print('pick_time: ', pick_time)
                                pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                                pif.write(f"{msg}\n")

                                # publish to mqtt (pick_msg)
                                message = {
                                    "station" : tmp[0],
                                    "channel" : tmp[1],
                                    "network" : tmp[2],
                                    "location" : tmp[3],
                                    "longitude" : tmp[4],
                                    "latitude" : tmp[5],
                                    "pa" : tmp[6],
                                    "pv" : tmp[7],
                                    "pd" : tmp[8],
                                    "tc" : tmp[9],
                                    "ptime" : tmp[10],
                                    "weight" : tmp[11],
                                    "instrument" : tmp[12],
                                    "upd_sec" : tmp[13],
                                }
                                client.publish(f"{local_env['PICK_MSG_TOPIC']}/{local_env['CHECKPOINT_TYPE']}", json.dumps(message))
                                
                        pif.close() 

                    # plotting the station on the map and send info to Line notify
                    cur_time = datetime.utcfromtimestamp(time.time())
                    print(f"{len(picked_coord)} stations are picked! <- {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                        
                else:
                    # plotting the station on the map and send info to Line notify
                    cur_time = datetime.utcfromtimestamp(time.time())
                    print(f"0 stations are picked! <- {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            else:
                print(f"(else)0 stations are picked! <- {cur}")  

            prev_key_index = cur_key_index

            # pending until 1 second
            while True:
                cur = time.time()
                if cur - system_record_time >= float(local_env['MIN_SECOND_ITERATION']):
                    break

            # avg: 0.2 s
            # _ = gc.collect()
        except Exception as e:
            # log the pending 
            # print(e)
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{local_env['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}.log"
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

# Calculating the Pa, Pv, and Pd per sample, only Z-component will be calculated
def Pavd_calculator(pavd_calc, waveform_comein, waveform_scnl, waveform_comein_length, pavd_sta, stationInfo, env_config):
    print('Starting Pavd_calculator...')
    MyManager.register('EEW_sta', EEW_sta)
    
    local_env = {}
    for k, v in env_config.items():
        local_env[k] = v

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
                if local_env['SOURCE'] == 'Palert' or local_env['SOURCE'] == 'CWASN' or local_env['SOURCE'] == 'TSMIP':
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
            picking_logfile = f"./log/exception/{local_env['CHECKPOINT_TYPE']}_{cur.year}-{cur.month}-{cur.day}.log"
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

# sending information to Pavd_calculator
def PavdModule_sender(CHECKPOINT_TYPE, SAMP_RATE, pavd_calc, waveform_comein, waveform_comein_length, pavd_scnl, waveform_scnl):
    
    # Send information into first Pavd module if idle
    while True:
        try:
            w = pavd_scnl.get()

            if w == {}:
                continue

            for k, v in w.items():
                scnl = k
                waveform = v

            for i in range(7):
                if pavd_calc[i].value == 0:
                    waveform_scnl[i].value = scnl
                    waveform_comein[i][0][:SAMP_RATE] = torch.from_numpy(waveform)
                    waveform_comein_length[i].value += SAMP_RATE   
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

class MQTT_DecisionMaking():
    def __init__(self,):
        self.activate_mqtt()

        self.env_config = {}
        for k, v in dotenv_values(sys.argv[1]).items():
            self.env_config[k] = v

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code " + str(rc))
    
            client.subscribe(f"{self.env_config['PICK_MSG_TOPIC']}/#")

            cur = datetime.fromtimestamp(time.time())
            with open(f"./log/mqtt/{cur.year}_{cur.month}_{cur.day}.log", 'a') as f:
                f.write(f"MQTT connected at {cur.hour}:{cur.minute}:{cur.second}\n")
                f.write('='*25)
                f.write('\n')
        else:
            print("Failed to connect, ", rc)
            client.disconnect()

    def on_message(self, client, userdata, msg):
        # parse the package: 0.0003 s on average
        msg = str(msg.payload.decode("utf-8"))
        msg = json.loads(msg)

        cur = time.time()
        # append to list for neighbor picking
        if cur - msg['ptime'] <= 5:
            pass

        
    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print("Unexpected disconnection: ", str(rc))
            # Optionally, add reconnection logic here
            client.reconnect()

            cur = datetime.fromtimestamp(time.time())
            with open(f"./log/mqtt/{cur.year}_{cur.month}_{cur.day}.log", 'a') as f:
                f.write(f"MQTT disconnected at {cur.hour}:{cur.minute}:{cur.second}\n")
                f.write('='*25)
                f.write('\n')

    def activate_mqtt(self):
        # 建立 MQTT Client 物件
        self.client = mqtt.Client()
        # 設定建立連線回呼函數 (callback function)
        self.client.on_connect = self.on_connect

        # callback functions
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect    

        # 連線至 MQTT 伺服器（伺服器位址,連接埠）, timeout=6000s
        self.client.connect(self.env_config['MQTT_SERVER'], int(self.env_config['PORT']), int(self.env_config['TIMEOUT']))
        print("connect!")
        
        # 進入無窮處理迴圈
        # self.client.loop_forever()
        self.client.loop_start()

    def init_shared_params(self):
        self.pick_msg = []

# creating directory while activating the system
def create_dir():
    if not os.path.exists('./log/picking'):
        os.makedirs('./log/picking')
    if not os.path.exists('./log/notify'):
        os.makedirs('./log/notify')
    if not os.path.exists('./log/original_picking'):
        os.makedirs('./log/original_picking')
    if not os.path.exists('./log/exception'):
        os.makedirs('./log/exception')
    if not os.path.exists('./log/statistical'):
        os.makedirs('./log/statistical')
    if not os.path.exists('./log/mqtt'):
        os.makedirs('./log/mqtt')
    if not os.path.exists('./plot'):
        os.makedirs('./plot')
    if not os.path.exists('./plot/trigger'):
        os.makedirs('./plot/trigger')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force = True)
    
    # create the directories
    create_dir()
      
    # start the system
    # including mqtt, picker, ... modules
    mqttserver_pred = Mqtt_Predictor()
    mqttserver_pred.start()
