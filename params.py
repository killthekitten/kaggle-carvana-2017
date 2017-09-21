import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--gpu')
arg('--fold', type=int, default=None)
arg('--n_folds', type=int, default=5)
arg('--folds_source')
arg('--seed', type=int, default=80)
arg('--test_size_float', type=float, default=0.1)
arg('--epochs', type=int, default=30)
arg('--img_height', type=int, default=1280)
arg('--img_width', type=int, default=1918)
arg('--out_height', type=int, default=1280)
arg('--out_width', type=int, default=1918)
arg('--input_width', type=int, default=1024)
arg('--input_height', type=int, default=1024)
arg('--use_crop', type=bool, default=True)
arg('--learning_rate', type=float, default=0.00001)
arg('--batch_size', type=int, default=1)
arg('--dataset_dir', default='input')
arg('--models_dir', default='models')
arg('--weights', default=None)
arg('--loss_function', default='boot_hard')
arg('--freeze_till_layer', default='input_1')
arg('--show_summary', type=bool)

arg('--pred_mask_dir', default='/home/selim/kaggle/datasets/carvana/predicted_masks/test_run')
arg('--pred_batch_size', default=1)
arg('--test_data_dir', default='/home/selim/kaggle/datasets/carvana/test')

# Dir names
arg('--train_data_dir_name', default='train')
arg('--val_data_dir_name', default='train')
arg('--train_mask_dir_name', default='train_masks')
arg('--val_mask_dir_name', default='train_masks')

args = parser.parse_args()
