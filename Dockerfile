FROM python:3.8

RUN mkdir /PickerModule
COPY . /PickerModule
WORKDIR /PickerModule

RUN \
pip install -r requirements.txt && \
pip3 install torch && \
cd seisbench  && \
pip install . 

RUN ["CUDA_VISIBLE_DEVICES=0", "python", "picker.py"]

