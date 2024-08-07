FROM python:3.8

RUN mkdir /PickerModule
COPY . /PickerModule
WORKDIR /PickerModule

RUN \
pip install --upgrade pip && \
pip install -r requirements.txt && \
pip3 install torch && \
cd seisbench  && \
pip install . 

# Set the CUDA_VISIBLE_DEVICES environment variable
# ENV CUDA_VISIBLE_DEVICES=0

# Select the environment file for each picker
# RUN ["python", "picker.py", "./env/EQT_env"]

CMD [ "python3" ]