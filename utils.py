from keras import backend as K
import os
import numpy as np
import time

from scipy import ndimage


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


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def encode_predictions(predicted_dir, output_path, threshold=127):
    f_submit = open(output_path, "w")
    f_submit.write('img,rle_mask\n')
    num_masks = 0
    print('Predicting RLE encoding masks ...')
    time_read = 0.0  # seconds
    time_rle = 0.0  # seconds
    time_stringify = 0.0  # seconds
    for f in os.listdir(predicted_dir):
        num_masks += 1
        mask_img_path = os.path.join(predicted_dir, f)
        img_name = mask_img_path.split("/")[-1]
        t0 = time.clock()
        mask_img = ndimage.imread(mask_img_path, mode='L')
        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        time_read += time.clock() - t0
        t0 = time.clock()
        rle = rle_encode(mask_img)
        f_submit.write("{},{}\n".format(img_name, rle_to_string(rle)))
        time_rle += time.clock() - t0
        t0 = time.clock()
        time_stringify += time.clock() - t0
    print('Time spent reading mask images:', time_read, 's, =>', 1000 * (time_read / num_masks), 'ms per mask.')
    print('Time spent RLE encoding masks:', time_rle, 's, =>', 1000 * (time_rle / num_masks), 'ms per mask.')
    print('Time spent stringifying RLEs:', time_stringify, 's, =>', 1000 * (time_stringify / num_masks), 'ms per mask.')
    f_submit.close()