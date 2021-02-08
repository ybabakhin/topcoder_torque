FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y  && \
        apt-get install -y build-essential python3.6 python3.6-dev python3-pip && \
        python3.6 -m pip install pip --upgrade && \
        python3.6 -m pip install wheel

RUN apt-get install -y libsndfile1 libgl1-mesa-glx wget curl unzip

RUN mkdir /code
COPY ./ /code
WORKDIR /code

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install pipenv==2020.11.15
RUN pipenv sync

WORKDIR /

RUN chmod +x /code/train.sh
RUN chmod +x /code/pred.sh

ENV TORCH_HOME="/code/pretrained_models/"
ENV PIPENV_PIPFILE="/code/Pipfile"

RUN mkdir -p /code/pretrained_models/checkpoints
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth -P /code/pretrained_models/checkpoints/
