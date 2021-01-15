FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y  && \
        apt-get install -y build-essential python3.6 python3.6-dev python3-pip && \
        python3.6 -m pip install pip --upgrade && \
        python3.6 -m pip install wheel

RUN apt-get install -y libsndfile1 libgl1-mesa-glx wget curl unzip

RUN mkdir /work
COPY . /work
WORKDIR /work

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install pipenv==2020.11.15
RUN pipenv sync

RUN chmod +x test.sh
RUN chmod +x train.sh

ENV TORCH_HOME="/wdata/pretrained_models/"
