#!/usr/bin/env bash

# Clean all data before training
rm -rf /wdata/*

# Make new directories
mkdir -p /wdata/lightning_logs/
mkdir -p /wdata/pretrained_models/hub/checkpoints

echo "Download pretrained weights"
wget http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-fda993c95.pth -P /wdata/pretrained_models/hub/checkpoints/
wget http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth -P /wdata/pretrained_models/hub/checkpoints/

echo "Split data into folds"
pipenv run python create_folds.py \
general.data_dir=$1 \
general.train_csv=/wdata/train_ground_truth_proc_5.csv \
general.leaky_validation=true

echo "Prepare the data"
pipenv run python prepare_data.py \
general.data_dir=$1

echo "Training the models"

echo "Model 1 out of 4"

pipenv run python train.py --multirun \
general.data_dir=$1 \
model.model_id=3 \
training.fold=0,1,2,3,4 \
model.architecture_name=se_resnext50_32x4d \
training.max_epochs=45 \
training.lr=1e-4 \
training.mixup=0.2 \
training.balancing=false \
training.augmentations=hard \
general.seed=17 \
model.crop_method=resize

echo "Model 2 out of 4"

pipenv run python train.py --multirun \
general.data_dir=$1 \
model.model_id=4 \
training.fold=0,1,2,3,4 \
model.architecture_name=dpn92 \
training.max_epochs=45 \
training.lr=1e-4 \
training.mixup=0.2 \
training.balancing=false \
training.augmentations=hard \
general.seed=19 \
model.crop_method=resize

echo "Model 3 out of 4"

pipenv run python train.py --multirun \
general.data_dir=$1 \
model.model_id=7 \
training.fold=0,1,2,3,4 \
model.architecture_name=se_resnext50_32x4d \
training.max_epochs=45 \
training.lr=1e-4 \
training.mixup=0.2 \
training.balancing=false \
training.augmentations=hard \
general.seed=25 \
training.pretrain_path='${general.logs_dir}model_3/fold_${training.fold}/' \
model.crop_method=resize

echo "Model 4 out of 4"

pipenv run python train.py --multirun \
general.data_dir=$1 \
model.model_id=8 \
training.fold=0,1,2,3,4 \
model.architecture_name=dpn92 \
training.max_epochs=45 \
training.lr=1e-4 \
training.mixup=0.2 \
training.balancing=false \
training.augmentations=hard \
general.seed=27 \
training.pretrain_path='${general.logs_dir}model_4/fold_${training.fold}/' \
model.crop_method=resize
