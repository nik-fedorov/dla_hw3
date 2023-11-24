# TTS

## Report

About all details about this homework can be found in 
[wandb report](https://wandb.ai/nik-fedorov/dla_hw3/reports/TTS--Vmlldzo2MDczMzY1). 

## Description

This is a repository containing a convenient pipeline for training text to speech models. 

Advantages of this repo:
- possibility of changing experimental configuration by only tuning one json file
- good and clean code structure (see `tts` folder with all elements of pipeline)
- prepared scripts for training and evaluation of models
- prepared downloadable checkpoint trained on LJSpeech dataset

## Installation guide

To set up the environment for this repository run the following command in your terminal (with your virtual environment activated):

```shell
pip install -r ./requirements.txt
```

## Evaluate model

To download my best checkpoint run the following:
```shell
python default_test_model/download_best_ckpt.py
```
if you are interested how I got this checkpoint, you can read about that in 
[wandb report](https://wandb.ai/nik-fedorov/dla_hw3/reports/TTS--Vmlldzo2MDczMzY1).

You can synthesize audio using `test.py` script. Here is an example of command to run my best checkpoint with default test config:

```shell
python test.py \
  -c default_test_model/config.json \
  -r default_test_model/checkpoint.pth \
  -t test_data/texts.txt \
  -o test_data
```

Moreover, you can synthesize audio for `test_data/texts.txt` with different durations, 
pitches and energies using command
```shell
python test_data/inference.py
```

## Training
Use `train.py` for training. Example of command to launch training from scratch:
```shell
python train.py -c tts/configs/train_config.json
```

To fine-tune your checkpoint you can use option `-r` to pass path to the checkpoint file:
```shell
python train.py \
  -c tts/configs/train_config.json \
  -r saved/models/<exp name>/<run name>/checkpoint.pth
```

## Credits

This repository is based on a [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.
