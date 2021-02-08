#!/usr/bin/env bash

if [ "$(ls -A /tmp/lightning_logs/)" ]; then
     echo "Trained weights available"
else
    echo "Loading pretrained weights"
fi

echo "Prepare the data"
pipenv run python /code/prepare_data.py \
general.data_dir=""

echo "Testing the models"

echo "Model 1 out of 3"

pipenv run python /code/test.py \
model.model_id=1 \
model.architecture_name=resnet18d \
testing.mode=test \
testing.n_slices=0 \
general.gpu_list=[]

echo "Model 2 out of 3"

pipenv run python /code/test.py \
model.model_id=2 \
model.architecture_name=tf_efficientnet_b1_ns \
testing.mode=test \
testing.n_slices=0 \
general.gpu_list=[]

echo "Model 3 out of 3"

pipenv run python /code/test.py --multirun \
model.model_id=3 \
model.architecture_name=resnet18d \
testing.mode=test \
testing.n_slices=0 \
general.gpu_list=[0] \
model.tabular_data=true

echo "Model Blending"

pipenv run python /code/blending.py \
ensemble.model_ids=[1,2,3] \
testing.mode=test \
general.gpu_list=[]
