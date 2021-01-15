#!/usr/bin/env bash

if [ "$(ls -A /wdata/lightning_logs/)" ]; then
     echo "Trained weights available"
else
    echo "Loading pretrained weights"
    mkdir -p /wdata/pretrained_models/hub/checkpoints
    wget http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-fda993c95.pth -P /wdata/pretrained_models/hub/checkpoints/
    wget http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth -P /wdata/pretrained_models/hub/checkpoints/

    mkdir -p /wdata/lightning_logs/
    fileId=1LF8AyqlWG8kc5yhoRrZyi2UwLafpi91Y
    fileName=/wdata/lightning_logs/bes_model_weights.zip
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
    code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

    unzip /wdata/lightning_logs/bes_model_weights.zip -d /wdata/lightning_logs/
fi

echo "Prepare the data"
pipenv run python prepare_data.py \
testing.test_data_dir=$1

echo "Testing the models"

echo "Models 1 out of 2"

pipenv run python test.py --multirun \
model.model_id=7 \
model.architecture_name=se_resnext50_32x4d \
model.input_size=[256,448] \
training.batch_size=1 \
testing.mode=test \
testing.test_data_dir=$1 \
testing.n_slices=3 \
general.gpu_list=[0,1,2,3] \
model.crop_method=resize

pipenv run python test.py --multirun \
model.model_id=7 \
model.architecture_name=se_resnext50_32x4d \
model.input_size=[256,2820] \
training.batch_size=32 \
testing.mode=test \
testing.test_data_dir=$1 \
testing.n_slices=0 \
general.gpu_list=[0,1,2,3] \
model.crop_method=resize

echo "Models 2 out of 2"

pipenv run python test.py --multirun \
model.model_id=8 \
model.architecture_name=dpn92 \
model.input_size=[256,448] \
training.batch_size=1 \
testing.mode=test \
testing.test_data_dir=$1 \
testing.n_slices=3 \
general.gpu_list=[0,1,2,3] \
model.crop_method=resize

pipenv run python test.py --multirun \
model.model_id=8 \
model.architecture_name=dpn92 \
model.input_size=[256,2820] \
training.batch_size=32 \
testing.mode=test \
testing.test_data_dir=$1 \
testing.n_slices=0 \
general.gpu_list=[0,1,2,3] \
model.crop_method=resize

pipenv run python blending.py \
testing.mode=test \
ensemble.model_ids=[7,8] \
testing.test_output_path=$2
