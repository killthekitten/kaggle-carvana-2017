# kaggle-carvana-2017

## Setup environment

Symlink your dataset folder (optional, you can pass an absolute path with `--dataset-dir`):
```
ln -s /path/to/dataset input
```

Install all the required packages:
```
pip install -r requirements.txt
```

## Run training

```
python train_unet_resnet.py\
  --gpu=0\
  --seed=42\
  --test_size_float=0.1\
  --dataset_dir='input'\
  --models_dir='../models/carvana/resnet_2'\
  --weights='weights/resnet-on-test-combined-19200.000010-0-0.0037752-99.6908383.h5'
```
