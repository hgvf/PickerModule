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
import joblib
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class MQTT_DecisionMaking():
    def __init__(self,):
        self.env_config = {}
        for k, v in dotenv_values(sys.argv[1]).items():
            self.env_config[k] = v

        self.init_shared_params()

        self.activate_mqtt()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code " + str(rc))
    
            checkpoint_type = ['phaseNet', 'REDPAN', 'GRADUATE', 'STALTA', 'eqt']

            for c in checkpoint_type:
                client.subscribe(f"{self.env_config['PICK_MSG_TOPIC']}/{c}")
                # print(f"{self.env_config['PICK_MSG_TOPIC']}/{c}")
            
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
        # print('msg coming: ', msg)
        cur = time.time()
        # append to list for neighbor picking
        if cur - float(msg['ptime']) <= int(self.env_config['TIME_COLLECT']):
            self.pick_msg.put(msg)
            self.system_time = time.time()
        
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
        print("MQTT decisionMaking connect!")
        
        # 進入無窮處理迴圈
        # self.client.loop_forever()
        self.client.loop_start()

    def init_shared_params(self):
        manager = Manager()
        self.pick_msg = manager.Queue()
        self.system_time = Value('d', int(time.time()))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        # get the station's info
        if self.env_config['SOURCE'] == 'Palert':
            self.stationInfo = get_PalertStationInfo(self.env_config['STATION_FILEPATH'])
        elif self.env_config['SOURCE'] == 'CWASN':
            self.stationInfo = get_CWBStationInfo(self.env_config['STATION_FILEPATH'])
        elif self.env_config['SOURCE'] == 'TSMIP':
            self.stationInfo = get_TSMIPStationInfo(self.env_config['STATION_FILEPATH'])
        else:
            self.stationInfo = get_StationInfo(self.env_config['STATION_FILEPATH'], (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))

        # parameters for notifier
        self.notify_TF = Value('d', int(0))
        self.toNotify_pickedCoord = manager.dict()
        self.n_notify = Value('d', int(0))
        # get the candidate line notify tokens
        self.notify_tokens, _ = load_tokens(self.env_config['NOTIFY_TOKENS'], self.env_config['WAVEFORM_TOKENS'])

    def start(self):
        try:
            decision_maker = Process(target=DecisionMaker, args=(self.env_config, self.pick_msg, self.system_time, self.stationInfo,
                                                                self.notify_TF, self.n_notify, self.toNotify_pickedCoord, self.device))
            decision_maker.start()

            notifier = Process(target=Notifier, args=(self.notify_TF, self.toNotify_pickedCoord, self.notify_tokens, self.n_notify,))
            notifier.start()

            decision_maker.join()
            notifier.join()
        except KeyboardInterrupt:
            decision_maker.terminate()
            decision_maker.join()

            notifier.terminate()
            notifier.join()

def DecisionMaker(env_config, pick_msg, system_time, stationInfo, notify_TF, n_notify, toNotify_pickedCoord, device):
    print("Starting Decision Maker")

    # localize the env_config dictionary
    local_env = {}
    for k, v in env_config.items():
        local_env[k] = v

    # Table for neighbor picking
    _, neighbor_table = station_selection(sel_chunk=local_env["CHUNK"], station_list=stationInfo, opt=local_env['SOURCE'], build_table=True, n_stations=int(local_env["N_PREDICTION_STATION"]), threshold_km=float(local_env['THRESHOLD_KM']),
                                            nearest_station=int(local_env['NEAREST_STATION']), option=local_env['TABLE_OPTION'])

    # ML classifier
    if local_env['DECISION_TYPE'] == 'ML':
        if local_env['DECISION_ML_CLASSIFIER'] == 'XGBoost':
            # model = XGBClassifier()
            # model.load_model(local_env['XGB_CLASSIFIER_PATH'])
            model = joblib.load(local_env['XGB_CLASSIFIER_PATH'])
        elif local_env['DECISION_ML_CLASSIFIER'] == 'RandomForest':
            # model = RandomForestClassifier()
            model = joblib.load(local_env['RF_CLASSIFIER_PATH'])
    elif local_env['DECISION_TYPE'] == 'DL':
        model = Ensemble(5, 21.901, 25.274, 120.0, 122, 32, False).to(device)
        checkpoint = torch.load(local_env['DL_CLASSIFIER_PATH'], map_location=device)
        model.load_state_dict(checkpoint['model'])

    # mqtt server for publishing pick_msg
    client = mqtt.Client()
    client.connect(env_config['MQTT_SERVER'], int(env_config['PORT']), int(env_config['TIMEOUT']))

    while True:
        try:
            if pick_msg.qsize() == 0:
                continue

            pick_sta = []
            pick_other = []
            
            start_parsing_package = time.time()
            while time.time()-start_parsing_package < float(local_env['SECOND_ACTIVATE_DECISION_MAKER']):
                while pick_msg.qsize() >= 1:
                    package = pick_msg.get()
                    
                    if int(package['weight']) <= int(local_env['REPORT_P_WEIGHT']):
                        pick_sta.append(package['station'])
                        pick_other.append(package)
                        start_parsing_package = time.time()
            
            print('pick_sta: ', pick_sta)
            # Neighbor picking
            if local_env['DECISION_TYPE'] == 'neighbor':
                original_res = [True for _ in range(len(pick_sta))]
                nei_res = neighbor_picking(neighbor_table, pick_sta, original_res, original_res, int(local_env['THRESHOLD_NEIGHBOR']))   # 用鄰近測站來 pick
                    
                # Ensemble
                en_res = ensemble_picking(pick_sta, int(local_env['THRESHOLD_N_PICKER']))

                # Merging the results from neighbor & ensemble picking
                res = np.logical_or(nei_res, en_res)

            else:
                toPredict = collect_classifier_data(pick_other)
                print(toPredict)
                if local_env['DECISION_TYPE'] == 'ML':
                    res = ML_classifier(model, toPredict)
                elif local_env['DECISION_TYPE'] == 'DL':
                    toPredict = torch.FloatTensor(toPredict).to(device)
                    out = model(toPredict).detach().cpu.numpy()
                    res = np.empty(out.shape)
                    res[out>=0.5] = True
                    res[out<0.5] = False

            # Publish the result to MQTT broker
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/picking/DecisionMaking_{cur.year}-{cur.month}-{cur.day}_picking_chunk.log"
            toPublish_package = np.array(pick_other)[res]

            # writing picking log file
            picked_coord = []
            record = []
            with open(picking_logfile,"a") as pif:
                cur_time = datetime.utcfromtimestamp(time.time())
                pif.write('='*25)
                pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                pif.write('='*25)
                pif.write('\n')

                for package in toPublish_package:
                    scnl = f"{package['station']}-{package['channel']}-{package['network']}-{package['location']}"
                    if scnl in record:
                        continue
                    else:
                        record.append(scnl)

                    pick_time = datetime.utcfromtimestamp(float(package['ptime']))
                    pif.write(f"{package['station']}_{package['channel']}_{package['network']}_{package['location']},\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                    pif.write(f"{package['station']}\t{package['channel']}\t{package['network']}\t{package['location']}\t")
                    pif.write(f"{package['longitude']}\t{package['latitude']}\t")
                    pif.write(f"{package['pa']}\t{package['pv']}\t{package['pd']}\t{package['tc']}\t")
                    pif.write(f"{package['ptime']}\t{package['weight']}\t{package['instrument']}\t{package['upd_sec']}\n")
                    
                    picked_coord.append((float(package['longitude']), float(package['latitude'])))

                    client.publish(f"{local_env['PICK_MSG_TOPIC']}/DecisionMaking", json.dumps(package))
            
                pif.close()

                # plotting the station on the map and send info to Line notify
            cur_time = datetime.utcfromtimestamp(time.time())
            print(f"{len(picked_coord)} stations are picked! <- {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")

            # send signal for Notifier to send Line notify
            if len(picked_coord) >= int(local_env["REPORT_NUM_OF_TRIGGER_DECISION_MAKING"]) and int(local_env['DECISION_MAKING_LINE_NOTIFY']) == 1:
                n_notify.value *= 0
                n_notify.value += len(picked_coord)
                notify_TF.value += 1
                for picked_idx in range(len(picked_coord)):
                    toNotify_pickedCoord[picked_idx] = picked_coord[picked_idx]

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/DecisionMaking_{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (DecisionMaker): {e}\n")
                pif.write(f"Trace back (DecisionMaker): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            continue

# notifing 
def Notifier(notify_TF, toNotify_pickedCoord, line_tokens, n_notify):
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
            trigger_plot_filename = f"DecisionMaking_{cur_time.year}-{cur_time.month}-{cur_time.day}_{cur_time.hour}_{cur_time.minute}_{cur_time.second}"
            
            start = time.time()
            line_token_number = plot_taiwan(trigger_plot_filename, picked_coord, line_tokens, line_token_number, 'DecisionMaking')

            notify_TF.value *= 0

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/DecisionMaking_{cur.year}-{cur.month}-{cur.day}.log"
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

    decisionMaking = MQTT_DecisionMaking()
    decisionMaking.start()