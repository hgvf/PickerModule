# Picker Module

## Step0: Cloning the repository
```shell=
$ git clone https://github.com/hgvf/PickerModule.git
```

## Step1: Setting the system configuation
* 系統參數檔 -> ```.env``` ([參數說明](env_parameters.md))
* (Optional) 指定 GPU number -> ```Dockerfile``` (CUDA_VISIBLE_DEVICES=<gpu_number>)

## Step2: Building and activating the picker module

```shell=
$ docker build -t picker:v1 .
```

## Step3: Activate the picker module
* Single picker
```shell=
$ TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=<gpu ID> python picker.py <path to env file> 
```

* Multiple pickers
  * 在實驗室的 mqtt 很常因為網路傳輸量過大，導致 mqtt connection lost，所以如果要同時用多個 pickers 就用這條指令
```shell=
$ TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=<gpu ID> python multi_picker.py ./env/DecisionMaking_env
```

* Decision making
```shell=
$ CUDA_VISIBLE_DEVICES=<gpu ID> python decision_making.py ./env/DecisionMaking_env

```
---
## TODO
* 要在 building image 時或是 之後再進入 container 啟動
* container volumn 問題，與 log 檔案有關
* 還要 Line notify?
* NVIDIA cuda launcher 每台都要安裝才能使用 GPU 運算
