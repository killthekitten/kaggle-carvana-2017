# kaggle-carvana-2017

## Setup environment

```
pip install -r requirements.txt
```

## Run training

```
python train_unet_resnet.py\
  --gpu=0\
  --dataset_dir='../datasets/carvana'\
  --models_dir='../models/carvana/resnet_2'\
  --weights='weights/resnet-on-test-combined-19200.000010-0-0.0037752-99.6908383.h5'
