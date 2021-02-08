#!/usr/bin/env bash

# Make new directories
mkdir -p /tmp/lightning_logs/

echo "Split data into folds"
pipenv run python /code/create_folds.py

echo "Prepare the data"
pipenv run python /code/prepare_data.py \
testing.test_data_dir=""

echo "Training the models"

echo "Model 1 out of 3"

pipenv run python /code/train.py --multirun \
model.model_id=1 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet18d \
training.max_epochs=45 \
training.augmentations=hard \
training.lr=1e-3 \
training.batch_size=32 \
model.tabular_data=false \
model.input_size=[256,512] \
model.dropout=0.1


echo "Model 2 out of 3"

pipenv run python /code/train.py --multirun \
model.model_id=2 \
training.fold=0,1,2,3,4 \
model.architecture_name=tf_efficientnet_b1_ns \
training.max_epochs=45 \
training.augmentations=hard \
training.lr=1e-3 \
training.batch_size=32 \
model.tabular_data=false \
model.input_size=[256,512] \
model.dropout=0.1


echo "Model 3 out of 3"

pipenv run python /code/train.py --multirun \
model.model_id=3 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet18d \
training.max_epochs=45 \
training.augmentations=hard \
training.lr=1e-3 \
training.batch_size=32 \
model.tabular_data=true \
model.input_size=[256,512] \
model.dropout=0.1
