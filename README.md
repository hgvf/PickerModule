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

---
## TODO
* 要在 building image 時或是 之後再進入 container 啟動
* container volumn 問題，與 log 檔案有關
* 還要 Line notify?
* NVIDIA cuda launcher 每台都要安裝才能使用 GPU 運算