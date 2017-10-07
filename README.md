# kaggle-carvana-2017

Team "80 TFlops" solution to [Carvana Image Masking Challenge on Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/). 4th place with TensorFlow-backed Keras.

Solution overview at Yandex (2017-10-07): [YouTube (Russian)](https://youtu.be/ilzq5huGr8U?t=17m56s), [pdf (English)](https://gh.mltrainings.ru/presentations/Mushinskiy_KaggleCarvanaImageMasking%20Challenge_2017.pdf).

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

## Run predict on train data

```
python predict_multithreaded.py \
  --gpu=0,1,2\
  --weights=weights/resnet-refine-fold_3-10240.000010-5-0.0033064-99.7107349.h5\
  --test_data_dir=input/train\
  --pred_mask_dir=predicted_masks/val_from_fold_3\
  --fold=3\
  --folds_source=folds.csv\
  --dataset_dir=input\
  --predict_on_val=True
```

or, alternatively

```
sh predict_train.sh
```
