# Parameters of picker module

```*```: 重要參數
```!```: Predefined 參數，須按照說明使用
```?```: Optional 參數，有需要再使用，或使用 default


## Section1: MQTT-related
| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| *```MQTT_SERVER``` | MQTT server's IP | 140.118.127.89 |
| *```PORT``` | port number | 1883 |
| ```TIMEOUT``` | MQTT 未收到訊息時，多久要 reconnect | 6000 (second) |
| *```WAVE_TOPIC``` | 帶有波形封包的 topic (prefix, 通常是地震網名稱) | RSD24Bit, TSMIP24Bit, CWASN24Bit | 
| *```PICK_MSG_TOPIC``` | picker publishing 使用的 topic | PICK24Bit |

---
## Section2: File path
| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| *```STATION_FILEPATH``` | 測站資訊檔案的路徑 | ./nsta/palertlist.csv | 
| ?```NOTIFY_TOKENS``` | Line notify 發報使用之 token 檔案 | ./nsta/notify_tokens.txt | 
| ?```WAVEFORM_TOKENS``` | Line notify 畫波形使用之 token 檔案 | ./nsta/waveform_tokens.txt | 
| *```PICKER_CHECKPOINT_PATH``` | 模型 trained weight 路徑 | ./checkpoints/EQT.pt |

---
## Section3: Data-related
* 系統會先將所有測站依照經緯度分區，每個 chunk 會有 ```N_PREDICTION_STATION``` 個測站，最後一個 chunk 則不一定
* 三種 ```CHUNK``` 使用方式
    1. ```CHUNK```=-1: 使用所有測站
    2. ```CHUNK```=0: 使用第 0 個 chunk 之測站
    3. ```CHUNK```=0-2: 使用第 0~2 個 chunks
* ```STORE_LENGTH```/```PREDICT_LENGTH```會根據使用的 picker 種類有變動
    * EQTransformer -> 6000/3000
    * GRADUATE -> 4000/2000
    * phaseNet -> 6002/3001

| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| *```CHUNK``` | 將所有測站切分數等分後，欲預測之 chunk | -1 (all chunks), 0 (specific chunk), 0-2 (multiple chunks) |
| *```SOURCE``` | 使用之地震網 | Palert, TSMIP, CWASN |
| !```STORE_LENGTH``` | 波形 buffer 長度 | 6000 (sample point) |
| !```PREDICT_LENGTH``` | 模型預測所需波形長度 | 3000 (sample point) |
| *```N_PREDICTION_STATION``` | 系統預測之測站數量 | 800 (stations) |
| ?```SAMP_RATE``` | sampling rate | 100 (Hz) |

---
## Section4: Picker-related
* 根據不同的 picker 會設定不同的 picking threshold (```THRESHOLD_TYPE```/```THRESHOLD_PROB```/```THRESHOLD_TRIGGER```)
    * ```CHECKPOINT_TYPE```=eqt -> max/0.5/x
    * ```CHECKPOINT_TYPE```=GRADUATE -> max/0.5/x
    * ```CHECKPOINT_TYPE```=phaseNet -> max/0.5/x
* 波形前處理，根據不同 picker 有不同流程 (```ZSCORE```/```FILTER```)
    * ```CHECKPOINT_TYPE```=eqt -> 1/1
    * ```CHECKPOINT_TYPE```=GRADUATE -> 1/1
    * ```CHECKPOINT_TYPE```=phaseNet -> 1/0
(filter 預設為 1-10Hz bandpass filter)

| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| !```CHECKPOINT_TYPE``` | picker 種類 | eqt, GRADUATE, phaseNet |
| !```THRESHOLD_TYPE``` | picking criteria | max, avg, continue |
| !```THRESHOLD_PROB``` | threshold of picking probability | 0.5 |
| !```THRESHOLD_TRIGGER``` | 搭配 picking crieria | 40 (sample) |
| !```ZSCORE``` | 波形是否需要 zscore normalization | 0 or 1 |
| !```FILTER``` | 波形是否需要 filter | 0 or 1 |

---
## Section5: AvoidFP-related
* 防誤報機制，每次選定 picked stations 時都要考慮鄰近測站的預測結果
    * Step1. 模型分別預測各測站的結果
    * Step2. If picked, 則套用防誤報機制
        * **方圓公里法**: 每次檢查測站方圓```THRESHOLD_KM```公里內所有測站，若有超過```THRESHOLD_NEIGHBOR```個測站同樣被 pick，則該測站會被選為 picked station
        * **鄰近測站法**(較快): 每次檢查距離測站最近的```NEAREST_STATION```個測站，若有超過```THRESHOLD_NEIGHBOR```個測站同樣被 pick，則該測站會被選為 picked station

| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| *```AVOID_FP``` | 是否啟用防誤報機制 |  0 or 1 |
| !```TABLE_OPTION``` | 防誤報機制種類 | nearest or km |
| !```THRESHOLD_KM``` | 蒐集每個測站方圓 x km 內的其他測站代碼 | 20 (km) |
| !```NEAREST_STATION``` | 蒐集距離每個測站最近的 x 站點 | 5 (station) |
| !```THRESHOLD_NEIGHBOR``` | 決定 picked station 時，鄰近 x stations 也要同時被 picked | 2 (station) |
| ?```PICK_GAP``` | 防止過多報告，測站距離上次 picking 要相隔 x second(s) | 1 (second) |

---
## Section6: Notify-related
* 為確保 picked station 資訊符合即時性，利用 ```VALID_PICKTIME``` 過濾
    * ```VALID_PICKTIME```=2500 -> 當 picking time 在 prediction window 的後```VALID_PICKTIME```個 sample 才算 (與```PREDICT_LENGTH``` 搭配使用)
    * ex. 確保 pick 在最近的五秒 = (```PREDICT_LENGTH```-```VALID_PICKTIME```)/```SAMP_RATE``` = 5

| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| *```REPORT_P_WEIGHT``` | P-weight 高於多少才算 picked | 1 (越低越具信心) |
| *```REPORT_NUM_OF_TRIGGER``` | 需有多少 picked stations 才發報 | 1 (station) |
| !```VALID_PICKTIME``` | 在多少 sample 內 pick 到才算 picked station | 2500 (sample) |
| *```LINE_NOTIFY``` | 是否需要 Line Notify | 0 or 1 |
| *```PLOT_WAVEFORM``` | 發報時是否需要同步將波形用 Line Notify 傳送 | 0 or 1 |

---
## Section 7: P's weight-related
* 兩種決定 P-weight 方法
    * ```PWEIGHT_TYPE```=prob -> 利用模型預測機率值
    * ```PWEIGHT_TYPE```=snr -> 利用波形 SNR 值 (signal & noise 分別為 picking time 後一秒與前一秒)

| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| ?```PWEIGHT_TYPE``` | 決定 P-weight 的根據 | prob or snr |
| ?```PWEIGHT0``` | 機率或 SNR 高於多少才歸類在 P-weight=0 | 0.7 (prob) or 20 (dB) |
| ?```PWEIGHT1``` | 機率或 SNR 高於多少才歸類在 P-weight=0 | 0.6 (prob) or 10 (dB) |
| ?```PWEIGHT2``` | 機率或 SNR 高於多少才歸類在 P-weight=0 | 0.5 (prob) or 5 (dB) |

---
## Section 8: System-related
* 每次啟動希望波形 buffer 不要都是空值，因此 picker 需要等待 buffer 都是有意義的波形才開始預測，通常與 ```PREDICT_LENGTH``` 搭配
    * ```SLEEP_SECOND``` = ```PREDICT_LENGTH```/```SAMP_RATE```

| Param | Meaning | Example |
|:--------:|:--------:|:--------:|
| !```SLEEP_SECOND``` | 系統剛啟動需等待多久才開始預測 | 30 (second) |
| ?```N_PAVD_PROCESS``` | 需要多少 process 平行計算測站之 Pa, Pv, Pd | 7 (process) |
| *```SECOND_MOVE_BUFFER``` | 每隔多久將 buffer 往前移動，避免新波形被覆蓋 | 5 (second) |
| ?```MIN_SECOND_ITERATION``` | 每個 picker iteration 至少要多久，避免重複預測相同波形 | 0.75 (second) |
| ?```DELETE_PICKINGLOG_DAY``` | 每隔幾天刪除 picking log 檔案 | 90 (day) |
| ?```DELETE_NOTIFYLOG_DAY``` | 每隔幾天刪除 notify log 檔案 | 90 (day) |



