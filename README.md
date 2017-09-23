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

## Run training with Resnet50 encoder

```
python train.py\
  --gpu=1\
  --seed=80\
  --dataset_dir='input'\
  --models_dir='models'\
  --network='resnet50'\
  --preprocessing_function='caffe'\
  --weights='weights/resnet-on-test-combined-8960.000010-0-0.0044540-99.7300751.h5'\
  --fold=1\
  --n_folds=5\
  --folds_source='folds.csv'\
  --val_mask_dir='gif_train_masks'\
  --train_mask_dir='gif_train_masks'
```

## Run training with MobileNet encoder

```
python train.py\
  --gpu=1\
  --seed=80\
  --dataset_dir='input'\
  --models_dir='models'\
  --network='mobilenet'\
  --preprocessing_function='tf'\
  --weights='weights/mobilenet-on-test-6400.000010-5-0.0054162-99.7713900.h5'\
  --fold=1\
  --n_folds=5\
  --folds_source='folds.csv'\
  --val_mask_dir='gif_train_masks'\
  --train_mask_dir='gif_train_masks'
```

## Run submission encoder

```
python generate_encoded_submission.py\
  --pred_mask_dir='predicted_masks/subfolder'\
  --submissions_dir='submissions'\
  --pred_threads=32
```
