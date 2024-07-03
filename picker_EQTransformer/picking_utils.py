import numpy as np
import time
import pandas as pd
import random
import torch
import requests
from datetime import datetime, timedelta
from staticmap import StaticMap, CircleMarker, Polygon, Line
from scipy import integrate
from math import sin, cos, sqrt, atan2, radians
import bisect


# get the station's factor, latitude, lontitude, starttime, and endtime
def get_StationInfo(nsta_path, starttime):
    a_time = starttime[:4] + starttime[5:7] + starttime[8:10]
    st_time = int(a_time)
        
    d = {}
    with open(nsta_path) as f:
        for line in f.readlines():
            l = line.strip().split()
            
            # check the station is expired
            if st_time >= int(l[-2]) and st_time <= int(l[-1]):
                key = f"{l[0]}_{l[8]}_{l[7]}_0{l[5]}"
                # key = f"{l[0]}_{l[8]}_TW_0{l[5]}"
                d[key] = [float(l[1]), float(l[2]), [float(l[9]), float(l[10]), float(l[11])], l[-2], l[-1]]
    # print(d)
    return d

# get the predicted station's latitude, lontitude, and factor
def get_coord_factor(key, stationInfo):    
    output = []
    sta = []
    flag = []

    for k in key:
        # 尋找相對應的測站
        tmp = k.split('_')
        if tmp[1][-1] == '1' or tmp[1][:2] == 'HL':
            channel = 'FBA'
        elif tmp[1][-1] == '4' or tmp[1][:2] == 'EH':
            channel = 'SP'
        elif tmp[1][-1] == '7' or tmp[1][:2] == 'HH':
            channel = 'BB'
        
        cur_k = f"{tmp[0]}_{channel}_{tmp[2]}_{tmp[3]}"
        sta.append(cur_k)
        try:
            info = stationInfo[cur_k]
            
            output.append(info[:-2])
            flag.append(True)
            # print(" ok ")
        except:
            output.append([-1, -1, [1, 1, 1]])
            flag.append(False)
            # print("no flag")
            continue
        
    return output, sta, flag

def get_PalertStationInfo(palert_path):
    df = pd.read_csv(palert_path)
    
    stationInfo = {}
    for i in df.iterrows():
        stationInfo[i[1]['station']] = [i[1]['lontitude'], i[1]['latitude'], 16.718]
    
    return stationInfo

def get_CWBStationInfo(cwb_path):
    with open(cwb_path, 'r') as f:
        sta_eew = f.readlines()
    
    stationInfo = {}
    for l in sta_eew:
        tmp = l.split(' ')
        
        stationInfo[tmp[0]] = [tmp[8], tmp[5], tmp[-2]]

    return stationInfo

def get_TSMIPStationInfo(tsmip_path):
    with open(tsmip_path, 'r') as f:
        sta_eew = f.readlines()

    stationInfo = {}
    for l in sta_eew:
        tmp = l.split()
        stationInfo[tmp[0]] = [tmp[5], tmp[4], tmp[-2]]

    return stationInfo

def get_Palert_CWB_coord(key, stationInfo):
    output = []
    station = []
    flag = []
    for k in key:
        sta = k.split('_')[0]

        try:
            output.append(stationInfo[sta])
            station.append(sta)
            flag.append(True)
            # print("flag")
        except:
            output.append([-1, -1, -1])
            station.append(-1)
            flag.append(False)
            # print("no flag")
            continue
        
    return output, station, flag

# load line tokens
def load_tokens(notify_path, waveform_path):
    with open(notify_path, 'r') as f:
        notify = f.readlines()

    with open(waveform_path, 'r') as f:
        waveform = f.readlines()

    notify = [n.strip() for n in notify]
    waveform = [n.strip() for n in waveform]
    return notify, waveform

# pick the p-wave according to the prediction of the model
def evaluation(pred, threshold_prob, threshold_trigger, threshold_type):
    # pred: 模型預測結果, (batch_size, wave_length)
    # print("pred",pred.shape)
    # 存每個測站是否 pick 到的結果 & pick 到的時間點
    pred_isTrigger = []
    pred_trigger_sample = []
    
    for i in range(pred.shape[0]):
        isTrigger = False
        
        if threshold_type == 'single':
            a = np.where(pred[i] >= threshold_prob, 1, 0)

            if np.any(a):
                c = np.where(a==1)
                isTrigger = True
                pred_trigger = c[0][0]
            else:
                pred_trigger = 0
                
        elif threshold_type == 'avg':
            a = pd.Series(pred[i])    
            win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

            c = np.where(win_avg >= threshold_prob, 1, 0)

            pred_trigger = 0
            if c.any():
                tri = np.where(c==1)
                pred_trigger = tri[0][0]-threshold_trigger+1
                isTrigger = True

        elif threshold_type == 'continue':
            a = np.where(pred[i] >= threshold_prob, 1, 0)
           
            a = pd.Series(a)    
            data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()
          
            if threshold_trigger in data:
                pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
                isTrigger = True
            else:
                pred_trigger = 0

        elif threshold_type == 'max':
            pred_trigger = np.argmax(pred[i]).item()
            # print("pred_trigger",pred_trigger)
            if pred[i][pred_trigger] >= threshold_prob:
                isTrigger = True
            else:
                pred_trigger = 0

        # 將 threshold 與 picking time 分開進行
        if isTrigger:
            # print("pick!!!")
            # 當 prediction 有過 threshold，則挑選最高的那個點當作 picking time
            pred_trigger = torch.argmax(pred[i]).item()
            
        pred_isTrigger.append(isTrigger)
        pred_trigger_sample.append(pred_trigger)
        
    return pred_isTrigger, pred_trigger_sample

# check the pred_trigger is in valid_picktime
def EEW_pick(res, pred_trigger, valid_picktime):
    new_res = []
    for idx, pred in enumerate(pred_trigger):
        if pred <= valid_picktime or not res[idx]:
            new_res.append(False)
        else:
            new_res.append(True)
    # print(new_res)
    return res, pred_trigger, new_res

# generate the Pa, Pv, Pd of picked stations
def picking_append_info(EEW_sta, toPredict_scnl, res, pred_trigger, wavelength, clear=False):
    Pa, Pv, Pd = [], [], []
    duration = []
    sampling_rate = 100.0

    for i in range(len(res)):
        # not picked
        if not res[i]:
            continue

        try:
            cur = EEW_sta[toPredict_scnl[i]]
            # cur_Pa, cur_Pv, cur_Pd = round(cur['pa'], 6), round(cur['pv'], 6), round(cur['pd'], 6)
            # cur_Pa, cur_Pv, cur_Pd = cur['pa'], cur['pv'], cur['pd']
            # cur_Pa, cur_Pv, cur_Pd = EEW_sta[toPredict_scnl[i]].get_Pavd()

            cur_duration = (wavelength - pred_trigger[i]) / sampling_rate
            cur_duration = int(round(cur_duration))

            Pa.append(round(cur['pa'], 6))
            Pv.append(round(cur['pv'], 6))
            Pd.append(round(cur['pd'], 6))
            duration.append(cur_duration)

            if clear:
                # EEW_sta[toPredict_scnl[i]].reset_pavd()
                cur['pa'], cur['pv'], cur['pd'] = 0, 0, 0

        except Exception as e:
            Pa.append(0.0)
            Pv.append(0.0)
            Pd.append(0.0)

            cur_duration = (wavelength - pred_trigger[i]) / sampling_rate
            cur_duration = int(round(cur_duration))
            duration.append(cur_duration)

    return Pa, Pv, Pd, duration

# generate the p's weight
def picking_p_weight_info(pred, res, Pweight_type, threshold, Zcomponent_trc=None, pred_trigger=None):
    p_weight = []
    for i in range(len(res)):
        # not picked
        if not res[i]:
            continue

        weight0, weight1, weight2 = threshold
        if Pweight_type == 'prob':
            value = torch.max(pred[i]).item()

        elif Pweight_type == 'snr':
            noise_start_idx = pred_trigger[i]-100 if pred_trigger[i]-100 >= 0 else 0
            signal_end_idx = pred_trigger[i]+100 if pred_trigger[i]+100 >= 0 else 2999
                
            signal = torch.mean(torch.pow(Zcomponent_trc[i][noise_start_idx:pred_trigger[i]], 2))
            noise = torch.mean(torch.pow(Zcomponent_trc[i][pred_trigger[i]:signal_end_idx], 2))
                
            value = signal/noise

        if value >= weight0:
            weight = 0
        elif value >= weight1 and value < weight0:
            weight = 1
        elif value >= weight2 and value < weight1:
            weight = 2
        else:
            weight = 3

        p_weight.append(weight)

    return p_weight

# Calculate the distance between two coordinates
def distance(coord1, coord2):
    R = 6373.0

    lon1, lat1 = radians(coord1[0]), radians(coord1[1])
    lon2, lat2 = radians(coord2[0]), radians(coord2[1])

    dlon, dlat = lon2-lon1, lat2-lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c
    return distance

# 區域型 picking
def post_picking(station_factor_coords, res, threshold_km):
    pick_idx = np.arange(len(res))[res]

    total_lon, total_lat = 0.0, 0.0
    for idx, pick_sta in enumerate(pick_idx):
        total_lon += station_factor_coords[pick_sta][0]
        total_lat += station_factor_coords[pick_sta][1]

    avg_lon, avg_lat = total_lon / len(pick_idx), total_lat / len(pick_idx)
    middle_coord = (avg_lon, avg_lat)

    real_picked_res = []
    for idx, pick_result in enumerate(res):
        if not pick_result:
            real_picked_res.append(False)
            continue

        dis = distance(middle_coord, (station_factor_coords[idx][0], station_factor_coords[idx][1]))

        if dis <= threshold_km:
            real_picked_res.append(True)
        else:
            real_picked_res.append(False)

    return real_picked_res

# 鄰近測站 picking 
def neighbor_picking(neighbor_table, station_list, res, original_res, threshold_neighbor):
    pick_sta = np.array(station_list)[original_res]
    real_res = []
    
    for idx, sta in enumerate(station_list):
        try:
            if not res[idx]:
                real_res.append(False)
                continue

            n_pick = len(set(pick_sta).intersection(set(neighbor_table[sta])))

            if (len(neighbor_table[sta]) < threshold_neighbor and n_pick > 1) or (n_pick >= threshold_neighbor):
                real_res.append(True)
            else:
                real_res.append(False)
        except:
            continue
            
    return real_res

# 建表: 先把每個測站方圓 threshold_km 內的所有測站都蒐集進 dict
def build_neighborStation_table(stationInfo, threshold_km=None, nearest_station=3, option='nearest'):
    table = {}
    
    for sta in stationInfo:
        table[sta[0]] = []
    
    if option == 'km':
        for outer_idx, sta1 in enumerate(stationInfo):
            nearest3_sta, nearest3_dis = [], []
            for inner_idx, sta2 in enumerate(stationInfo):
                if inner_idx <= outer_idx:
                    continue

                dis = distance((sta1[1][0], sta1[1][1]), (sta2[1][0], sta2[1][1]))
                if dis <= threshold_km:
                    table[sta1[0]].append(sta2[0])
                    table[sta2[0]].append(sta1[0])
    
    elif option == 'nearest':
        for outer_idx, sta1 in enumerate(stationInfo):
            nearest3_sta, nearest3_dis = [], []
            for inner_idx, sta2 in enumerate(stationInfo):
                if sta1[0] == sta2[0]:
                    continue
                    
                dis = distance((sta1[1][0], sta1[1][1]), (sta2[1][0], sta2[1][1]))
                nearest3_sta.append(sta2[0])
                nearest3_dis.append(dis)

            nearest3_sta = np.array(nearest3_sta)
            nearest3_dis = np.array(nearest3_dis)
            table[sta1[0]] = nearest3_sta[np.argsort(nearest3_dis)[:nearest_station]].tolist()
            
    return table

# 檢查 picked station 在 picktime_gap 秒內有沒有 pick 過了，有的話先 ignore
def check_duplicate_pick(res, toPredict_scnl, pick_record, pred_trigger, waveform_starttime, pick_gap, STORE_LENGTH, PREDICT_LENGTH):
    pick_idx = np.arange(len(res))[res]
    sampling_rate = 100.0

    # 每次 picking 都會有誤差，跟上次 picking time 要差距超過 error_threshold 秒才能算是新地震
    error_threshold_sec = pick_gap
    for i in pick_idx:
        # convert pred_trigger into absolute datetime
        pick_sample = pred_trigger[i]
        sec, microsec = pick_sample // sampling_rate, pick_sample % sampling_rate
        
        # in picker, we take last 3000 samples waveforms
        # +8 hours is because .timestamp() will minus 8 hours automatically
        buffer_start_time_offset = STORE_LENGTH // 2 - PREDICT_LENGTH
        buffer_start_time_offset_sec, buffer_start_time_offset_microsec = buffer_start_time_offset // sampling_rate, buffer_start_time_offset % sampling_rate
        p_arrival_time = waveform_starttime + timedelta(seconds=float(sec), microseconds=float(microsec)*10000) + \
                                            timedelta(seconds=float(buffer_start_time_offset_sec), microseconds=float(buffer_start_time_offset_microsec)*10000)
        
        # if the station is picked in first time
        if toPredict_scnl[i] not in pick_record:
            pick_record[toPredict_scnl[i]] = p_arrival_time
        # if the station is picked before, check the previous picking time is PICK_TIME_GAP seconds ago
        else:
            prev_pickingtime = pick_record[toPredict_scnl[i]]
            if (p_arrival_time > prev_pickingtime) and (p_arrival_time - prev_pickingtime).seconds >= error_threshold_sec:
                pick_record[toPredict_scnl[i]] = p_arrival_time
            else:
                res[i] = False
            
    return res, pick_record

def gen_new_pred_trigger(res, pred_trigger, p_weight, p_weight_threshold):
    new_pred_trigger = []
    pick_idx = np.arange(len(res))[res]
    for idx, pp in enumerate(pick_idx):
        if p_weight[idx] <= p_weight_threshold:
            new_pred_trigger.append(pred_trigger[pp])
    
    return new_pred_trigger

# generate the picking message 
# Format: S C N L Longitude Latitude Pa Pv Pd Tc/PGA P-arrival P-weight Instrument/Src Duration
# H024 HLZ TW -- 120.204309 22.990450 2.549266 0.059743 0.039478 0.000000 1671078524.54000 2 1 3
def gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, waveform_starttime, Pavd, duration, P_weight, STORE_LENGTH, PREDICT_LENGTH, sampling_rate=100):
    pick_msg = []
    
    # grep the idx of station which is picked
    pick_idx = np.arange(len(res))[res]

    for idx, pick_sta in enumerate(pick_idx):
        # station info is unknown
        if station_factor_coords[pick_sta][0] == -1:
            continue
            
        cur_pickmsg = ""

        # append scnl into pick_msg
        scnl = toPredict_scnl[pick_sta].split('_')
        
        # convert channel: 1-9 to HL (1-3), EH (4-6), HH (7-9)
        if scnl[1][-1] == '1':
            channel = 'HLZ'
        elif scnl[1][-1] == '4':
            channel = 'EHZ'
        elif scnl[1][-1] == '7':
            channel = 'HHZ'
        else:
            channel = scnl[1]

        cur_pickmsg += f"{scnl[0]} {channel} {scnl[2]} {scnl[3]}"
        
        # append coordinate into pick_msg
        cur_pickmsg += f" {station_factor_coords[pick_sta][0]} {station_factor_coords[pick_sta][1]}"

        # Pa, Pv, and Pd
        # Tc/PGA is useless, fixed at 0.0
        cur_pickmsg += f" {Pavd[0][idx]} {Pavd[1][idx]} {Pavd[2][idx]} 0.0"
        
        # append p_arrival time into pick_msg
        pick_sample = pred_trigger[pick_sta]
        sec, microsec = pick_sample // sampling_rate, pick_sample % sampling_rate
        
        # in picker, we take last 3000 samples waveforms
        buffer_start_time_offset = STORE_LENGTH // 2 - PREDICT_LENGTH
        buffer_start_time_offset_sec, buffer_start_time_offset_microsec = buffer_start_time_offset // sampling_rate, buffer_start_time_offset % sampling_rate

        p_arrival_time = waveform_starttime + timedelta(seconds=float(sec), microseconds=float(microsec)*10000) + \
                                            timedelta(seconds=float(buffer_start_time_offset_sec), microseconds=float(buffer_start_time_offset_microsec)*10000)
        cur_pickmsg += f' {p_arrival_time.timestamp()}'

        # p_weight
        cur_pickmsg += f' {P_weight[idx]}'

        # fixed instrument with 1: acceleration
        # fixed upd_sec with 3: always use 3 seconds after p_arrival to calculate the Pa, Pv, and Pd
        cur_pickmsg += f' 4 {duration[idx]}'
     
        pick_msg.append(cur_pickmsg)
        
    return pick_msg

# plot the picking info on Taiwan map
def plot_taiwan(name, coords1, token, token_number, CHECKPOINT_TYPE):
    m = StaticMap(300, 400)

    for sta in coords1:
        marker = CircleMarker(sta, '#eb4034', 8)

        m.add_marker(marker)

    image = m.render(zoom=7)
    image.save(f"./plot/trigger/{name}.png")

    token_number = random.sample(range(len(token)), k=1)[0]
    token_number = notify(len(coords1), name, token, token_number, CHECKPOINT_TYPE)
    
    return token_number

# send the picking info to Line notify
def notify(n_sta, name, token, token_number, CHECKPOINT_TYPE):
    message = '\n(Palert)' + str(CHECKPOINT_TYPE) + '\n'+str(n_sta) + ' 個測站偵測到 P 波\n報告時間: '+str((datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))
    message += '\n'

    cnt = 0
    while True:
        if token_number >= len(token):
            token_number = len(token) - 1

        cnt += 1
        if cnt >= 3:
            break   

        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token[token_number]}'
            }
            payload = {
                'message': message,
            }
            image = {
                'imageFile': open(f"./plot/trigger/{name}.png", 'rb'),
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
                files=image,
            )
            if response.status_code == 200:
                print(f"Success coordination -> {response.text}")
                break
            else:
                print(f'(Notify) Error -> {response.status_code}, {response.text}')
                token_number = random.sample(range(len(token)), k=1)[0]
        except Exception as e:
            print(e)
            token_number = random.sample(range(len(token)), k=1)[0]
    return token_number

# send the picking info to Line notify
def plot_notify(name, token, token_number, CHECKPOINT_TYPE, SOURCE):    
    msg = name.split('/')[-1].split('.')[0] 
    message = f"({SOURCE}) {CHECKPOINT_TYPE} Prediction: {msg}\n"

    cnt = 0
    while True:
        if token_number >= len(token):
            token_number = len(token) - 1

        cnt += 1
        if cnt >= 3:
            break
         
        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token[token_number]}'
            }
            payload = {
                'message': message,
            }
            image = {
                'imageFile': open(name, 'rb'),
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
                files=image,
            )
            if response.status_code == 200:
                print(f"Success waveform prediction -> {response.text}")
                break
            else:
                print(f'(Waveform) Error -> {response.status_code}, {response.text}')
                token_number = random.sample(range(len(token)), k=1)[0]
        except Exception as e:
            print(e)
            token_number = random.sample(range(len(token)), k=1)[0]
    return token_number

# sent the notify proved the system is alive
def alive_notify(token, token_number, CHECKPOINT_TYPE, SOURCE):
    message = f"({SOURCE}) {CHECKPOINT_TYPE} sysgem is alive: \n"
    message += (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f')

    cnt = 0
    while True:
        if token_number >= len(token):
            token_number = len(token) - 1
            
        cnt += 1
        if cnt >= 3:
            break

        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token[token_number]}'
            }
            payload = {
                'message': message,
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
            )
            if response.status_code == 200:
                print(f"Success, system is alive -> {response.text}")
                break
            else:
                print(f'(Alive) Error -> {response.status_code}, {response.text}')
                token_number = random.sample(range(len(token)), k=1)[0]
        except Exception as e:
            print(e)
            token_number = random.sample(range(len(token)), k=1)[0]
    return token_number

# select the stations to collect the waveform
def station_selection(sel_chunk, station_list, opt, build_table=False, n_stations=None, threshold_km=None,
                    nearest_station=3, option='nearest'):
    if opt == 'CWB':
        lon_split = np.array([120.91])
        lat_split = np.array([21.913 , 22.2508961, 22.5887922, 22.9266883, 23.2645844,
        23.6024805, 23.9403766, 24.2782727, 24.6161688, 24.9540649,
        25.291961 ])
        
        # 依照經緯度切分測站
        chunk = [[] for _ in range(40)]
        for k, sta in station_list.items():
            row, col = 0, 0

            row = bisect.bisect_left(lon_split, float(sta[0]))
            col = bisect.bisect_left(lat_split, float(sta[1]))
            
            chunk[2*col+row].append((k, [float(sta[0]), float(sta[1]), float(sta[2][0]), float(sta[2][1]), float(sta[2][2])]))

        # 微調前步驟的結果，使每個區域都不超過 55 個測站
        output_chunks = []
        output_chunks.append(chunk[5]+chunk[4]+chunk[3]+chunk[2]+chunk[0])
        output_chunks.append(chunk[6])
        output_chunks.append(chunk[7]+chunk[9]+chunk[11])
        output_chunks.append(chunk[13]+chunk[15])
        output_chunks.append(chunk[14])
        output_chunks.append(chunk[16]+chunk[18])
        output_chunks.append(chunk[17])
        
        chunk[19] = sorted(chunk[19], key = lambda x : x[1][1])
        output_chunks.append(chunk[19][len(chunk[19])//2:])
        output_chunks.append(chunk[19][:len(chunk[19])//2])
        
        tmp_chunk = chunk[8]+chunk[10]+chunk[12]
        tmp_chunk = sorted(tmp_chunk, key = lambda x : x[1][1])
        output_chunks.append(tmp_chunk[:len(tmp_chunk)//4])
        output_chunks.append(tmp_chunk[len(tmp_chunk)//4:len(tmp_chunk)//4 * 2])
        output_chunks.append(tmp_chunk[len(tmp_chunk)//4 * 2:len(tmp_chunk)//4 * 3])
        output_chunks.append(tmp_chunk[len(tmp_chunk)//4 * 3:])
        
        chunk[21] = sorted(chunk[21], key = lambda x : x[1][1])
        output_chunks.append(chunk[21][len(chunk[21])//3:len(chunk[21])//3 * 2])
        output_chunks.append(chunk[21][len(chunk[21])//3 * 2:len(chunk[21])//3 * 3])
        output_chunks.append(chunk[21][len(chunk[21])//3 * 3:])
    
        # if sel_chunk == -1, then collect all station in TSMIP
        if sel_chunk == -1:
            output_chunks = []
            for k, v in station_list.items():
                output_chunks.append((k, [float(v[0]), float(v[1]), float(v[2][0]), float(v[2][1]), float(v[2][2])]))
            output_chunks = [output_chunks]

        table = 0
        if build_table:
            # build the table that contains every station in "threshold_km" km for each station
            table = build_neighborStation_table(output_chunks[sel_chunk], threshold_km, nearest_station, option)

        output = []
        if sel_chunk != -1:
            for o in output_chunks[sel_chunk]:
                output.append(o[0])
        else:
            for ii in range(len(output_chunks)):
                for o in output_chunks[ii]:
                    output.append(o[0])
        print('Partial station list: ', len(output))
        return output, table

    elif opt == 'TSMIP':
        lon_split = np.array([120.91])
        lat_split = np.array([21.9009 , 24.03495, 26.169  ])

        # 依照經緯度切分測站
        chunk = [[] for _ in range(6)]
        for k, sta in station_list.items():
            row, col = 0, 0

            row = bisect.bisect_left(lon_split, float(sta[0]))
            col = bisect.bisect_left(lat_split, float(sta[1]))

            chunk[2*col+row].append((k, [float(sta[0]), float(sta[1]), float(sta[2])]))

        output_chunks = []
        output_chunks.append(chunk[3])
        
        chunk[2] = sorted(chunk[2], key = lambda x : x[1][1])
        output_chunks.append(chunk[2][len(chunk[2])//2:])
        output_chunks.append(chunk[2][:len(chunk[2])//2])
        output_chunks[-1] += chunk[0]
        
        chunk[5] = sorted(chunk[5], key = lambda x : x[1][0])
        output_chunks.append(chunk[5][:50] + chunk[4])
        output_chunks.append(chunk[5][50:])

        new_output_chunks2 = []
        for sta in output_chunks[2]:
            if sta[1][1] <= 22.5:
                output_chunks[0].append(sta)
            else:
                new_output_chunks2.append(sta)
        output_chunks[2] = new_output_chunks2

        new_output_chunks1 = []
        for sta in output_chunks[1]:
            if sta[1][1] >= 23.977:
                output_chunks[3].append(sta)
            else:
                new_output_chunks1.append(sta)
        output_chunks[1] = new_output_chunks1

        # if sel_chunk == -1, then collect all station in TSMIP
        if sel_chunk == -1:
            output_chunks = []
            for k, v in station_list.items():
                output_chunks.append((k, [float(v[0]), float(v[1]), float(v[2])]))
            output_chunks = [output_chunks]
       
        table = 0
        if build_table:
            # build the table that contains every station in "threshold_km" km for each station
            table = build_neighborStation_table(output_chunks[sel_chunk], threshold_km, nearest_station, option)
                
        output = []
        for o in output_chunks[sel_chunk]:
            output.append(o[0])

        return output, table

    else:
        # sort the station_list by lontitude
        stationInfo = sorted(station_list.items(), key = lambda x : x[1][0])

        if sel_chunk == -1:
            n_stations = len(stationInfo)

        station_chunks = [stationInfo[n_stations*i:n_stations*i+n_stations] 
                            for i in range(len(stationInfo)//n_stations)]
        station_chunks += [stationInfo[n_stations*(len(stationInfo)//n_stations):]]

        table = 0
        if build_table:
            # build the table that contains every station in "threshold_km" km for each station
            table = build_neighborStation_table(station_chunks[sel_chunk], threshold_km, nearest_station, option)

        return [i[0] for i in station_chunks[sel_chunk]], table

# For TSMIP, 用 station_code 代碼再分區
def ForTSMIP_station_selection(stationInfo, target_length):
    sta = []
    must_contain = ['J', 'I', 'H', 'G', 'E', 'F']       # 這些分區中的測站全數保留
    total_length = len(stationInfo)
    downsampling_rate = target_length / total_length

    print('Before selection, station: ', len(stationInfo))
    for s in stationInfo.keys():
        if s[0] in must_contain:
            sta.append(s)

    region = ['A', 'B', 'C', 'D']
    a, b, c, d = [], [], [], []
    for s in stationInfo.keys():
        if s[0] == 'A':
            a.append(s)
        elif s[0] == 'B':
            b.append(s)
        elif s[0] == 'C':
            c.append(s)
        elif s[0] == 'D':
            d.append(s)
    
    sta += random.sample(a,int(len(a)*downsampling_rate))
    sta += random.sample(b,int(len(b)*downsampling_rate))
    sta += random.sample(c,int(len(c)*downsampling_rate))
    sta += random.sample(d,int(len(d)*downsampling_rate))
    print('To predict station: ', len(sta))

    # 分區結果記錄下來並上傳到 google drive
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")
    drive = GoogleDrive(gauth)

    filepath = 'TSMIP_station_selction.log'
    with open(filepath, 'a') as f:
        for s in sta:
            f.write(f"{s}\n")
        f.close()

    # file1 = drive.CreateFile({"title":filepath,"parents": [{"kind": "drive#fileLink", "id": "1SClOfjAZH2x6Ei693kSNtiWqnzvz0gHm"}]})
    # file1.SetContentFile(filepath)
    # file1.Upload() #檔案上傳
    print("Result of TSMIP station selection -> uploading succeeded!")

    return sta