# 7th place solution to Topcoder's "Predict the torque from the bolt tightening sound" challenge
[Competition website](https://www.topcoder.com/challenges/07596fc0-961b-471b-aca9-0932501ef594)

In this problem, one is given an audio file containing the sound made while bolting steel material on a construction site. The code should read the audio file and predict the torque value of it.

## Instructions to run the code

### System Requirements
The following system requirements should be satisfied:
* OS: Ubuntu 16.04
* Python: 3.6
* CUDA: 10.1
* cudnn: 7
* Docker

### Environment Setup
1. Build a docker image: `docker build -t topcoder .`
2. Start a docker container:
```
docker run --rm --gpus all --shm-size 16G \
-v /path/to/train/audios:/data/input/train \
-v /path/to/train/labels:/data/gt/train \
-v /path/to/save/models:/tmp \
-v /path/to/test/audios:/data/input/pred \
-v /path/to/test/predictions:/data/output/pred \
-it topcoder
```

### Build the Model
1. Use `/code/train.sh` to train the model
1. Use `/code/test.sh` to make the inference
