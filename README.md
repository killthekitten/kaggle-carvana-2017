# kaggle-carvana-2017

## Setup environment

Symlink your dataset folder (optional, you can pass an absolute path with `--dataset-dir`):
```
ln -s /path/to/dataset input
```

All of the data is expected to be in that folder, either as a file in the root or as a subdirectory.

Install all the required packages:
```
pip install -r requirements.txt
```

You should have Keras master installed:
```
pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps
```

## Run training

```
python train_unet_resnet.py\
  --gpu=1\
  --seed=80\
  --dataset_dir='input'\
  --models_dir='models'\
  --weights='weights/resnet-on-test-combined-8960.000010-0-0.0044540-99.7300751.h5'\
  --fold=1\
  --n_folds=5\
  --folds_source='folds.csv'\
  --val_mask_dir='gif_train_masks'\
  --train_mask_dir='gif_train_masks'
```
