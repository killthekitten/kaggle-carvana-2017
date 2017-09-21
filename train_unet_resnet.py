import os

from keras.losses import binary_crossentropy

from keras_iterator import ImageDataGenerator
from losses import make_loss, dice_coef_clipped, dice_coef
from models import get_unet_resnet
from random_transform_mask import ImageWithMaskFunction

import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


img_height = 1280
img_width = 1918
out_height = 1280
out_width = 1918
input_height = 1024
input_width = 1024
use_crop = True
learning_rate = 0.00001
batch_size = 1
nbr_train_samples = 4576
nbr_validation_samples = 512
freeze_till_layer = "input_1"
nbr_epochs = 30
dataset_dir = '/home/selim/kaggle/datasets/carvana'
mask_dir = os.path.join(dataset_dir, "train_masks")
val_mask_dir = os.path.join(dataset_dir, "train_masks")
models_dir = '/home/selim/kaggle/models/carvana/resnet_2'
best_model_file = models_dir + "/resnet-refine-" + str(input_width) + format(learning_rate, ".6f") + "-{epoch:d}-{val_loss:0.7f}-{val_dice_coef_clipped:0.7f}.h5"
train_data_dir = os.path.join(dataset_dir, 'train_split_2')
val_data_dir = os.path.join(dataset_dir, 'train_val_2')
weights = "weights/resnet-on-test-combined-19200.000010-0-0.0037752-99.6908383.h5"
loss_function = "boot_hard"
def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index]:
            l.trainable = False

def preprocess_input_resnet(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


def preprocess_input(x):
    return preprocess_input_resnet(x)

model = get_unet_resnet((input_height, input_width, 3))

freeze_model(model, freeze_till_layer)
if weights is not None:
    model.load_weights(weights)
optimizer = Adam(lr=learning_rate)
model.summary()
model.compile(loss=make_loss(loss_function), optimizer=optimizer, metrics=[dice_coef, binary_crossentropy, dice_coef_clipped])

model.summary()

crop_size = None
if use_crop:
    crop_size = (input_height, input_width)
mask_function = ImageWithMaskFunction(out_size=(out_height, out_width), crop_size=crop_size, mask_dir=val_mask_dir)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    classes=None,
    class_mode='regression',
    output_function=ImageWithMaskFunction(out_size=(out_height, out_width), crop_size=crop_size, mask_dir=mask_dir).mask_pred_train)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    classes=None,
    class_mode='regression', output_function=mask_function.mask_pred_val)

best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=nbr_train_samples / batch_size + 1,
    epochs=nbr_epochs,
    validation_data=val_generator,
    validation_steps=nbr_validation_samples / batch_size + 1,
    callbacks=[best_model, EarlyStopping(patience=45, verbose=10)], workers=2)
