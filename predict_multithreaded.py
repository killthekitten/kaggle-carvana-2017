import os

import numpy as np
import tensorflow as tf
import pandas as pd
from datasets import generate_filenames
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import array_to_img, load_img, img_to_array
from tensorflow.python.client import device_lib

from models import make_model
from params import args
import threading
import queue

gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']

prediction_dir = args.pred_mask_dir

output_dir = args.pred_mask_dir
batch_size = args.pred_batch_size

if args.predict_on_val:
    folds_df = pd.read_csv(os.path.join(args.dataset_dir, args.folds_source))
    train_ids = generate_filenames(folds_df[folds_df.fold != args.fold]['id'])
    val_ids = generate_filenames(folds_df[folds_df.fold == args.fold]['id'])
    ids = val_ids
else:
    ids = sorted(os.listdir(args.test_data_dir))

filenames = [os.path.join(args.test_data_dir, f) for f in ids]

q_size = 10


def create_model(gpu):
    with tf.device(gpu):
        model = make_model((None, None, 3))
    model.load_weights(args.weights)
    return model


def data_loader(q, ):
    for start in range(0, len(filenames), batch_size):
        x_batch = []
        end = min(start + batch_size, len(filenames))
        filenames_batch = filenames[start:end]

        for filename in filenames_batch:
            img = img_to_array(load_img(filename))
            x_batch.append(img)

        x_batch = preprocess_input(np.array(x_batch, np.float32), mode=args.preprocess_input)
        padded_x = np.zeros((batch_size, 1280, 1920, 3))
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
