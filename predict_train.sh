python predict_multithreaded.py \
  --gpu=0,1,2\
  --weights=weights/resnet-refine-fold_4-10240.000010-9-0.0034898-99.6993431.h5\
  --test_data_dir=input/train\
  --pred_mask_dir=predicted_masks/val_from_fold_4\
  --fold=4\
  --folds_source=folds.csv\
  --dataset_dir=input\
  --predict_on_val=True
python predict_multithreaded.py \
  --gpu=0,1,2\
  --weights=weights/resnet-refine-fold_1-10240.000010-9-0.0033331-99.7088822.h5\
  --test_data_dir=input/train\
  --pred_mask_dir=predicted_masks/val_from_fold_1\
  --fold=1\
  --folds_source=folds.csv\
  --dataset_dir=input\
  --predict_on_val=True
python predict_multithreaded.py \
  --gpu=0,1,2\
  --weights=weights/resnet-refine-fold_2-10240.000010-10-0.0032741-99.7180628.h5\
  --test_data_dir=input/train\
  --pred_mask_dir=predicted_masks/val_from_fold_2\
  --fold=2\
  --folds_source=folds.csv\
  --dataset_dir=input\
  --predict_on_val=True
python predict_multithreaded.py \
  --gpu=0,1,2\
  --weights=weights/resnet-refine-fold_3-10240.000010-5-0.0033064-99.7107349.h5\
  --test_data_dir=input/train\
  --pred_mask_dir=predicted_masks/val_from_fold_3\
  --fold=3\
  --folds_source=folds.csv\
  --dataset_dir=input\
  --predict_on_val=True
python predict_multithreaded.py \
  --gpu=0,1,2\
  --weights=weights/resnet-refine-fold_0-10240.000010-12-0.0032666-99.7145138.h5\
  --test_data_dir=input/train\
  --pred_mask_dir=predicted_masks/val_from_fold_0\
  --fold=0\
  --folds_source=folds.csv\
  --dataset_dir=input\
  --predict_on_val=True
