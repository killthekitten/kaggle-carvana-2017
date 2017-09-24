import os

import numpy as np
import tensorflow as tf
import pandas as pd
from datasets import generate_filenames
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import array_to_img, load_img, img_to_array, flip_axis
from tensorflow.python.client import device_lib

from models import make_model
from params import args
import threading
import queue
from tqdm import tqdm

gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']

prediction_dir = args.pred_mask_dir

output_dir = args.pred_mask_dir
batch_size = args.pred_batch_size

if args.predict_on_val:
    folds_df = pd.read_csv(os.path.join(args.dataset_dir, args.folds_source))
    ids = generate_filenames(folds_df['id'])
else:
    test_images = set(os.listdir(args.test_data_dir))
    already_tested_images = set(map(lambda x: x.replace('.png', '.jpg'), os.listdir(output_dir)))
    ids = sorted(list(test_images - already_tested_images))

filenames = [os.path.join(args.test_data_dir, f) for f in ids]

q_size = 10

def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(x, 2)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(pred, 2)
    else:
        return pred

def create_model(gpu):
    with tf.device(gpu):
        model = make_model((None, None, args.stacked_channels + 3))
    model.load_weights(args.weights, by_name=True)
    return model


def data_loader(q, ):
    for start in tqdm(range(0, len(filenames), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(filenames))
        filenames_batch = filenames[start:end]

        for filename in filenames_batch:
            img = load_img(filename)

            stacked_channels = []
            for i in range(args.stacked_channels):
                channel_path = os.path.join(args.stacked_channels_dir,
                                            str(i),
                                            filename.split('/')[-1].replace('.jpg', '.png'))
                stacked_channel = load_img(channel_path, grayscale=True)
                stacked_channels.append(stacked_channel)
            stacked_img = np.dstack((img, *stacked_channels))

            x_batch.append(img_to_array(stacked_img))


        x_batch = preprocess_input(np.array(x_batch, np.float32), mode=args.preprocessing_function)
        if args.pred_tta:
            x_batch = do_tta(x_batch, args.pred_tta)
        padded_x = np.zeros((batch_size, 1280, 1920, args.stacked_channels + 3))
        padded_x[:, :, 1:-1, :] = x_batch
        q.put((filenames_batch, padded_x))

    for gpu in gpus:
        q.put((None, None))


def predictor(q, gpu):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        model = create_model(gpu)
        while True:
            batch_fnames, x_batch = q.get()
            if x_batch is None:
                break

            preds = model.predict_on_batch(x_batch)

            if args.pred_tta:
                preds = undo_tta(preds, args.pred_tta)

            for i, pred in enumerate(preds):
                filename = batch_fnames[i]
                prediction = pred[:, 1:-1, :]
                array_to_img(prediction * 255).save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))

print('Predicting on {} samples with batch_size = {}...'.format(len(filenames), batch_size))
q = queue.Queue(maxsize=q_size)
threads = []
threads.append(threading.Thread(target=data_loader, name='DataLoader', args=(q,)))
threads[0].start()
for gpu in gpus:
    print("Starting predictor at device " + gpu)

    t = threading.Thread(target=predictor, name='Predictor', args=(q, gpu))
    threads.append(t)
    t.start()
