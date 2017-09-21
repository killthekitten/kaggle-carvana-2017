import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from datasets import build_batch_generator, bootstrapped_split, generate_filenames
from losses import make_loss, dice_coef_clipped, dice_coef
from models import get_unet_resnet
from params import args
from utils import freeze_model
from sklearn.model_selection import KFold
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    mask_dir = os.path.join(args.dataset_dir, args.train_mask_dir_name)
    val_mask_dir = os.path.join(args.dataset_dir, args.val_mask_dir_name)

    train_data_dir = os.path.join(args.dataset_dir, args.train_data_dir_name)
    val_data_dir = os.path.join(args.dataset_dir, args.val_data_dir_name)

    # @TODO: add clipped `val_dice` to the filename
    best_model_file =\
        '{}/resnet-refine-{}{:.6f}'.format(args.models_dir, args.input_width, args.learning_rate) +\
        '-{epoch:d}-{val_loss:0.7f}-{val_dice_coef_clipped:0.7f}.h5'

    model = get_unet_resnet((None, None, 3))
    freeze_model(model, args.freeze_till_layer)

    if args.weights is not None:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights)
    else:
        print('No weights passed, training from scratch')

    optimizer = Adam(lr=args.learning_rate)

    if args.show_summary:
        model.summary()

    model.compile(loss=make_loss(args.loss_function),
                  optimizer=optimizer,
                  metrics=[dice_coef, binary_crossentropy, dice_coef_clipped])

    if args.show_summary:
        model.summary()

    crop_size = None

    if args.use_crop:
        crop_size = (args.input_height, args.input_width)
        print('Using crops of shape ({}, {})'.format(args.input_height, args.input_width))
    else:
        print('Using full size images, --use_crop=True to do crops')

    folds_df = pd.read_csv(os.path.join(args.dataset_dir, args.folds_source))
    train_ids = generate_filenames(folds_df[folds_df.fold != args.fold]['id'])
    val_ids = generate_filenames(folds_df[folds_df.fold == args.fold]['id'])

    train_generator = build_batch_generator(
        train_ids,
        img_dir=train_data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        mask_dir=mask_dir,
        aug=True
    )

    val_generator = build_batch_generator(
        val_ids,
        img_dir=val_data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        out_size=(args.out_height, args.out_width),
        crop_size=None,
        mask_dir=val_mask_dir,
        aug=False
    )

    best_model = ModelCheckpoint(best_model_file, monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=False,
                                                  save_weights_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_ids) / args.batch_size + 1,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_ids) / args.batch_size + 1,
        callbacks=[best_model, EarlyStopping(patience=45, verbose=10)],
        workers=1)

if __name__ == '__main__':
    main()
