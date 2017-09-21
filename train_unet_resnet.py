import os

from keras.losses import binary_crossentropy

from keras_iterator import ImageDataGenerator
from losses import make_loss, dice_coef_clipped, dice_coef
from models import get_unet_resnet
from random_transform_mask import ImageWithMaskFunction

import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from params import args

from utils import freeze_model, preprocess_input

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    # @TODO: infer automatically from the list of train/val indices
    nbr_train_samples = 4576
    nbr_validation_samples = 512

    mask_dir = os.path.join(args.dataset_dir, 'train_masks')
    val_mask_dir = os.path.join(args.dataset_dir, 'train_masks')
    best_model_file =\
        '{}/resnet-refine-{}{:.6f}'.format(args.models_dir, args.input_width, args.learning_rate) +\
        '-{epoch:d}-{val_loss:0.7f}-{val_dice_coef_clipped:0.7f}.h5'

    # @TODO: change to use common data dir with a list of train/val indices
    train_data_dir = os.path.join(args.dataset_dir, 'train_split_2')
    val_data_dir = os.path.join(args.dataset_dir, 'train_val_2')

    model = get_unet_resnet((args.input_height, args.input_width, 3))
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

    mask_function = ImageWithMaskFunction(out_size=(args.out_height, args.out_width),
                                          crop_size=crop_size,
                                          mask_dir=val_mask_dir)

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
        shuffle=True,
        classes=None,
        class_mode='regression',
        output_function=ImageWithMaskFunction(out_size=(args.out_height, args.out_width),
                                              crop_size=crop_size,
                                              mask_dir=mask_dir).mask_pred_train)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
        shuffle=True,
        classes=None,
        class_mode='regression', output_function=mask_function.mask_pred_val)

    best_model = ModelCheckpoint(best_model_file, monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=False,
                                                  save_weights_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nbr_train_samples / args.batch_size + 1,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=nbr_validation_samples / args.batch_size + 1,
        callbacks=[best_model, EarlyStopping(patience=45, verbose=10)], workers=2)

if __name__ == '__main__':
    main()
