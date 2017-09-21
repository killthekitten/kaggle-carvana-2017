import random

import numpy as np
import pandas as pd
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc.pilutil import imread

from params import args
from sklearn.model_selection import train_test_split
from random_transform_mask import ImageWithMaskFunction
import os


def pad(image, padding_w, padding_h):
    height, width, depth = image.shape
    new_image = np.zeros((height + padding_h * 2, width + padding_w * 2, depth), dtype=image.dtype)
    new_image[padding_h:(height + padding_h), padding_w:(width + padding_w)] = image
    # @TODO: Fill padded zones
    # new_image[:, :padding_w] = image[:, :padding_w]
    # new_image[:padding_h, :] = image[:padding_h, :]
    # new_image[-padding_h:, :] = image[-padding_h:, :]

    return new_image


def unpad(image, padding_w):
    return image[:, padding_w:(image.shape[1] - padding_w), :]


def generate_filenames(car_ids):
    return ['{}_{}'.format(id, str(angle + 1).zfill(2)) for angle in range(16) for id in car_ids]


def bootstrapped_split(car_ids, seed=args.seed):
    """
    # Arguments
        metadata: metadata.csv provided by Carvana (should include
        `train` column).

    # Returns
        A tuple (train_ids, test_ids)
    """
    all_ids = pd.Series(car_ids)
    train_ids, valid_ids = train_test_split(car_ids, test_size=args.test_size_float,
                                            random_state=seed)

    np.random.seed(seed)
    bootstrapped_idx = np.random.random_integers(0, len(train_ids))
    bootstrapped_train_ids = train_ids[bootstrapped_idx]

    return generate_filenames(bootstrapped_train_ids.values), generate_filenames(valid_ids)


def build_batch_generator(filenames, img_dir=None, batch_size=None,
                          shuffle=False, transformations=None,
                          out_size=None, crop_size=None, mask_dir=None, aug=False):
    mask_function = ImageWithMaskFunction(out_size=out_size, crop_size=crop_size, mask_dir=mask_dir)

    def batch_generator():
        while True:
            # @TODO: Should we fixate the seed here?
            if shuffle:
                random.shuffle(filenames)

            for start in range(0, len(filenames), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(filenames))
                train_batch = filenames[start:end]

                for filename in train_batch:
                    img = imread(os.path.join(img_dir, '{}.jpg'.format(filename)))
                    x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32)

                    batch_x, mask = mask_function.mask_pred(x_batch, filenames[start:end], range(batch_size), aug)
                    yield preprocess_input(batch_x, mode="caffe"), mask
    return batch_generator
