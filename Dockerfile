FROM python:3.8

RUN mkdir /PickerModule
COPY . /PickerModule
WORKDIR /PickerModule

RUN \
apt-get update -y && \
apt-get install python3-pip -y && \
git clone https://github.com/hgvf/PickerModule.git && \
pip install -r requirements.txt && \
pip3 install torch && \
cd seisbench  && \
pip install .  && \

RUN ["CUDA_VISIBLE_DEVICES=0", "python", "picker.py"]

